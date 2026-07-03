from flax import linen as nn
import jax

from .ssm import quantize_adc


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
            if self.glu_rank > 0:
                # Low-rank factorization: Dense(H, r) -> Dense(r, H).
                # Params: H*r + r*H + H = 2*H*r + H  (vs 2H*H + H = 2H^2 + H for full).
                # Bias only on the final projection; the down-projection is bias-free
                # since the gate is applied via sigmoid.
                self.out2_down = nn.Dense(self.glu_rank, use_bias=False)
                self.out2_up = nn.Dense(self.d_model)
            else:
                self.out2 = nn.Dense(self.d_model)

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
            if self.glu_rank > 0:
                # Low-rank gate: sigmoid(out2_up(out2_down(gelu(x))))
                gate_raw = self.out2_up(self.out2_down(x1))
            else:
                gate_raw = self.out2(x1)
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
