import math

from flax import linen as nn
import jax
import jax.numpy as jnp

from .ssm import quantize_adc


def _fanin_normal(fan_in, scale=1.0):
    """Fan-in-scaled normal init (lecun-style, std = scale/sqrt(fan_in)).
    Used for the structured-gate factors so each factor is ~unit-gain and the
    R-head Monarch sum keeps the pre-sigmoid variance O(1) (scale=1/sqrt(R) on
    the second factor).  Tune `scale` if the structured gate is unstable early.
    """
    std = scale / math.sqrt(fan_in)

    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * std

    return init


class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
            glu_rank    (int32):    if > 0 and activation is half_glu*, factorize
                                    the out2 Dense(H, H) into Dense(H, r) @ Dense(r, H)
                                    to sparsify the gate.  Default 0 = full-rank.
                                    Used for iso-param comparisons where main gate
                                    capacity is redistributed elsewhere (bigger SSM state).
    """
    ssm: nn.Module
    dropout: float
    d_model: int
    activation: str = "gelu"
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    glu_rank: int = 0
    # v2 structured gate (half_glu2 only).  "dense" = the v1 path (byte-identical:
    # Dense(H,r)@Dense(r,H) when glu_rank>0, else Dense(H,H)).  "monarch" / "blockdiag"
    # replace the out2 product with a structured low-rank operator (glu_rank is then
    # only the "gate-on" trigger; the structured op's size is set by the knobs below).
    glu_structure: str = "dense"
    glu_monarch_heads: int = 3     # R independent Monarch(b,m) ops summed; params = R*H*(b+m)
    glu_blockdiag_blocks: int = 2  # B diagonal blocks of (H/B)x(H/B); params = H^2/B + H
    # Per-layer DAC/ADC enable bools set by StackedEncoderModel based on
    # crossings_every.  Passed to SSM at instantiation; SSM uses them to
    # skip DAC-in / ADC-out quantization when this layer is INSIDE a
    # multi-layer analog chunk.  Defaults True (single-crossing baseline).
    dac_in_enabled: bool = True
    adc_out_enabled: bool = True
    # Digital-domain quantization: real chip stores digital tensors in
    # SRAM at limited precision (INT8/INT16), not FP32.  When > 0,
    # quantize gate matmul output and gate multiply result to
    # digital_bits precision.  0 = FP32 (idealized, matches original).
    digital_bits: int = 0

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        # Pass per-layer dac/adc enable bools to the SSM.  The partial's
        # baked-in defaults (True) get overridden by these kwargs since
        # functools.partial merges call-time kwargs with priority.
        self.seq = self.ssm(step_rescale=self.step_rescale,
                            dac_in_enabled=self.dac_in_enabled,
                            adc_out_enabled=self.adc_out_enabled)

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            if self.glu_structure == "monarch":
                self._setup_monarch()
            elif self.glu_structure == "blockdiag":
                self._setup_blockdiag()
            elif self.glu_structure == "dense":
                if self.glu_rank > 0:
                    # Low-rank factorization: Dense(H, r) -> Dense(r, H).
                    # Params: H*r + r*H + H = 2*H*r + H  (vs 2H*H + H = 2H^2 + H for full).
                    # Bias only on the final projection; the down-projection is bias-free
                    # since the gate is applied via sigmoid.
                    self.out2_down = nn.Dense(self.glu_rank, use_bias=False)
                    self.out2_up = nn.Dense(self.d_model)
                else:
                    self.out2 = nn.Dense(self.d_model)
            else:
                raise NotImplementedError(
                    f"glu_structure {self.glu_structure!r} not in "
                    "{'dense','monarch','blockdiag'}")

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name='batch')
        else:
            self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    # ---- v2 structured gate (half_glu2) -------------------------------------
    def _monarch_bm(self):
        """Factor H = b * m with b the largest power of two <= sqrt(H) dividing H.
        For H=128 -> (b, m) = (8, 16): both clean {8,16} systolic contraction dims."""
        H = self.d_model
        b = 1
        while (2 * b) * (2 * b) <= H and H % (2 * b) == 0:
            b *= 2
        m = H // b
        assert b * m == H, f"Monarch requires b*m==H; got b={b} m={m} H={H}"
        return b, m

    def _setup_monarch(self):
        b, m = self._monarch_bm()
        R = self.glu_monarch_heads
        H = self.d_model
        # Two block-diagonal factors + a transpose (the Monarch permutation), summed
        # over R heads.  Contraction dims are the block sizes {m, b} -> clean util.
        # Params: R*(b*m*m) + R*(m*b*b) + H = R*H*(b+m) + H.
        self.monarch_W1 = self.param("monarch_W1", _fanin_normal(m), (R, b, m, m))
        self.monarch_W2 = self.param("monarch_W2",
                                     _fanin_normal(b, 1.0 / math.sqrt(R)),
                                     (R, m, b, b))
        self.monarch_bias = self.param("monarch_bias", nn.initializers.zeros, (H,))

    def _apply_monarch(self, x):
        b, m = self._monarch_bm()
        R = self.glu_monarch_heads
        H = self.d_model
        L = x.shape[0]
        x0 = x.reshape(L, b, m)
        y1 = jnp.einsum("lbm,rbmn->rlbn", x0, self.monarch_W1)   # (R,L,b,m)
        y1t = jnp.swapaxes(y1, 2, 3)                              # (R,L,m,b) -- permutation
        y2 = jnp.einsum("rlmb,rmbc->rlmc", y1t, self.monarch_W2)  # (R,L,m,b)
        return y2.reshape(R, L, H).sum(axis=0) + self.monarch_bias

    def _setup_blockdiag(self):
        B = self.glu_blockdiag_blocks
        H = self.d_model
        assert H % B == 0, f"blockdiag requires d_model % blocks == 0; got H={H} B={B}"
        blk = H // B
        # B independent dense (blk x blk) blocks; contraction dim = blk (clean).
        # Params: B*blk*blk + H = H^2/B + H.  Block-local (no cross-block mixing).
        self.bd_W = self.param("bd_W", _fanin_normal(blk), (B, blk, blk))
        self.bd_bias = self.param("bd_bias", nn.initializers.zeros, (H,))

    def _apply_blockdiag(self, x):
        B = self.glu_blockdiag_blocks
        H = self.d_model
        blk = H // B
        L = x.shape[0]
        xb = x.reshape(L, B, blk)
        yb = jnp.einsum("lbi,bij->lbj", xb, self.bd_W)           # (L,B,blk)
        return yb.reshape(L, H) + self.bd_bias

    def _gate_raw(self, x1):
        """Compute pre-sigmoid gate for half_glu2, dispatching on glu_structure.
        'dense' is the v1 path (byte-identical)."""
        if self.glu_structure == "monarch":
            return self._apply_monarch(x1)
        if self.glu_structure == "blockdiag":
            return self._apply_blockdiag(x1)
        if self.glu_rank > 0:
            return self.out2_up(self.out2_down(x1))
        return self.out2(x1)

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)

        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            gate_raw = self.out2(x)
            if self.digital_bits > 0:
                gate_raw = quantize_adc(gate_raw, self.digital_bits)
            x = self.out1(x) * jax.nn.sigmoid(gate_raw)
            if self.digital_bits > 0:
                x = quantize_adc(x, self.digital_bits)
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            gate_raw = self.out2(x)
            if self.digital_bits > 0:
                gate_raw = quantize_adc(gate_raw, self.digital_bits)
            x = x * jax.nn.sigmoid(gate_raw)
            if self.digital_bits > 0:
                x = quantize_adc(x, self.digital_bits)
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            # gate = sigmoid(<gate op>(gelu(x))); dispatch dense / monarch / blockdiag.
            gate_raw = self._gate_raw(x1)
            if self.digital_bits > 0:
                gate_raw = quantize_adc(gate_raw, self.digital_bits)
            x = x * jax.nn.sigmoid(gate_raw)
            if self.digital_bits > 0:
                x = quantize_adc(x, self.digital_bits)
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(nn.gelu(x))
            if self.digital_bits > 0:
                x = quantize_adc(x, self.digital_bits)
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        x = skip + x
        if not self.prenorm:
            x = self.norm(x)
        return x
