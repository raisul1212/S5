import jax
import jax.numpy as np
from flax import linen as nn
from .layers import SequenceLayer


class StackedEncoderModel(nn.Module):
    """ Defines a stack of S5 layers to be used as an encoder.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                     we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    d_model: int
    n_layers: int
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    glu_rank: int = 0
    glu_structure: str = "dense"       # v2 gate: dense | monarch | blockdiag
    glu_monarch_heads: int = 3
    glu_monarch_b: int = 0
    glu_monarch_residual_rank: int = 0
    glu_blockdiag_blocks: int = 2
    # Chip architecture knob: how many analog SSM layers between DAC/ADC
    # boundary crossings.  1 = ADC+DAC every layer (baseline, 2L crossings
    # total).  N = fewer boundaries, deeper analog stacks (2*ceil(L/N)).
    # Applies only when the SSM is in chip-analysis mode (sigma>0 or
    # bits>0); at sigma=0 bits=0 the SSM fast path ignores these bools.
    crossings_every: int = 1
    # Digital-domain quantization for gate/multiply outputs.  Real chip
    # stores digital tensors at INT8/INT16, not FP32.  0 = FP32 (idealized).
    digital_bits: int = 0

    def setup(self):
        """
        Initializes a linear encoder and the stack of S5 layers.
        Per-layer dac_in_enabled / adc_out_enabled bools are computed
        from crossings_every: within each chunk of `crossings_every`
        contiguous layers, only the FIRST has a DAC-in boundary and only
        the LAST has an ADC-out boundary.  Layer 0 is forced to have
        DAC (embedding output is digital) and layer L-1 is forced to
        have ADC (classifier is digital) -- this covers the edge case
        where n_layers % crossings_every != 0.
        """
        self.encoder = nn.Dense(self.d_model)
        layers = []
        chunk_end_pos = self.crossings_every - 1
        for i in range(self.n_layers):
            chunk_pos = i % self.crossings_every
            dac_in = (chunk_pos == 0) or (i == 0)
            adc_out = (chunk_pos == chunk_end_pos) or (i == self.n_layers - 1)
            layers.append(SequenceLayer(
                ssm=self.ssm,
                dropout=self.dropout,
                d_model=self.d_model,
                activation=self.activation,
                training=self.training,
                prenorm=self.prenorm,
                batchnorm=self.batchnorm,
                bn_momentum=self.bn_momentum,
                step_rescale=self.step_rescale,
                glu_rank=self.glu_rank,
                glu_structure=self.glu_structure,
                glu_monarch_heads=self.glu_monarch_heads,
                glu_monarch_b=self.glu_monarch_b,
                glu_monarch_residual_rank=self.glu_monarch_residual_rank,
                glu_blockdiag_blocks=self.glu_blockdiag_blocks,
                dac_in_enabled=dac_in,
                adc_out_enabled=adc_out,
                digital_bits=self.digital_bits,
            ))
        self.layers = layers

    def __call__(self, x, integration_timesteps):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output sequence (float32): (L, d_model)
        """
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return x


def masked_meanpool(x, lengths):
    """
    Helper function to perform mean pooling across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length.
    Args:
         x (float32): input sequence (L, d_model)
         lengths (int32):   the original length of the sequence before padding
    Returns:
        mean pooled output sequence (float32): (d_model)
    """
    L = x.shape[0]
    mask = np.arange(L) < lengths
    return np.sum(mask[..., None]*x, axis=0)/lengths


# Here we call vmap to parallelize across a batch of input sequences
batch_masked_meanpool = jax.vmap(masked_meanpool)


class ClassificationModel(nn.Module):
    """ S5 classificaton sequence model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, a linear decoder, and a softmax operation.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            mode        (str):      Options: [pool: use mean pooling, last: just take
                                                                       the last state]
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    padded: bool
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    glu_rank: int = 0
    glu_structure: str = "dense"       # v2 gate: dense | monarch | blockdiag
    glu_monarch_heads: int = 3
    glu_monarch_b: int = 0
    glu_monarch_residual_rank: int = 0
    glu_blockdiag_blocks: int = 2
    crossings_every: int = 1  # chip arch knob -- see StackedEncoderModel
    digital_bits: int = 0     # chip digital precision -- see SequenceLayer

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.encoder = StackedEncoderModel(
                            ssm=self.ssm,
                            d_model=self.d_model,
                            n_layers=self.n_layers,
                            activation=self.activation,
                            dropout=self.dropout,
                            training=self.training,
                            prenorm=self.prenorm,
                            batchnorm=self.batchnorm,
                            bn_momentum=self.bn_momentum,
                            step_rescale=self.step_rescale,
                            glu_rank=self.glu_rank,
                            glu_structure=self.glu_structure,
                            glu_monarch_heads=self.glu_monarch_heads,
                            glu_monarch_b=self.glu_monarch_b,
                            glu_monarch_residual_rank=self.glu_monarch_residual_rank,
                            glu_blockdiag_blocks=self.glu_blockdiag_blocks,
                            crossings_every=self.crossings_every,
                            digital_bits=self.digital_bits,
                                        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x, integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            x, length = x  # input consists of data and prepadded seq lens

        x = self.encoder(x, integration_timesteps)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            if self.padded:
                x = masked_meanpool(x, length)
            else:
                x = np.mean(x, axis=0)

        elif self.mode in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError("Mode must be in ['pool'] for self.padded=True (for now...)")
            else:
                x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


# Here we call vmap to parallelize across a batch of input sequences
BatchClassificationModel = nn.vmap(
    ClassificationModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None, "intermediates": 0},
    split_rngs={"params": False, "dropout": True, "noise": True}, axis_name='batch')


class LMModel(nn.Module):
    """Autoregressive character-LM head: the stacked encoder (CAUSAL when the SSM
    is unidirectional) with NO pooling, then a per-position linear decoder over the
    vocab. Drop-in with the classification model's (x, integration_timesteps) call
    signature so all of train.py's setup (HiPPO, opt, checkpointing) is reused.

    Causality requirements (enforced by the caller, not here):
      - the SSM must be unidirectional (bidirectional=False),
      - batchnorm=False (LayerNorm) -- BatchNorm(axis_name='batch') normalizes each
        feature over batch x time and would leak future statistics into position t.
    """
    ssm: nn.Module
    d_output: int      # vocab size
    d_model: int
    n_layers: int
    padded: bool = False        # LM chunks are fixed-length; kept for call-signature parity
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    mode: str = ""              # unused (no pooling); kept for parity with model_cls kwargs
    prenorm: bool = False
    batchnorm: bool = False     # MUST be False for a causal LM (see docstring)
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    glu_rank: int = 0           # >0 = low-rank half_glu2 gate (e.g. 40 = corner3')
    glu_structure: str = "dense"

    def setup(self):
        self.encoder = StackedEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            glu_rank=self.glu_rank,
            glu_structure=self.glu_structure,
        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x, integration_timesteps):
        x = self.encoder(x, integration_timesteps)   # (L, d_model) -- no pooling
        x = self.decoder(x)                           # (L, vocab)
        return nn.log_softmax(x, axis=-1)             # (L, vocab) per-position log-probs


# vmap over the batch, exactly like BatchClassificationModel.
BatchLMModel = nn.vmap(
    LMModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None, "intermediates": 0},
    split_rngs={"params": False, "dropout": True, "noise": True}, axis_name='batch')


class MambinoLMModel(nn.Module):
    """Mambino-LM Stage 1: a 2-level dual-process predictive-coding LM head.

    Bottom (fast, per-token) = the causal StackedEncoderModel -> per-position
    logits p0.  Top (slow, deliberate) = a SECOND encoder that ticks once per
    `stride` tokens over the mean-POOLED bottom features (a subsampled/compressed
    long-range stream a bigger bottom state cannot cheaply represent).  A surprise
    gate on the bottom's REALIZED TRAILING error z0 decides, per token, whether to
    escalate (add the top's nudge) -- adaptive compute:

        logits(t) = p0(t) + alpha(t) * beta * nudge(t)

    alpha(t): SOFT sigmoid during training (gradients stay alive), HARD threshold
    at eval (the discrete skip the chip does).  This is the Stage-1 mechanism
    validated on CPU (paper_v2_ppac/mambino_lm/stage1_hierarchy.py): soft-train /
    hard-eval avoids the STE saturation that collapses the gate.

    CAUSALITY (all enforced here):
      - z0(t) is a trailing EMA of bottom errors e(0..t-1) ONLY (excludes e(t));
        e(tau)=CE(p0(tau), x[tau+1]) uses inputs up to x[tau+1]<=x[t], never x[t+1].
      - nudge(t) uses the top's summary through window w(t)-1 (a one-window shift):
        every token in window w reads only bottom features from windows < w.
      - the gate DECISION detaches z0 (stop-grad), matching the Cluster-A house
        rule; beta/nudge get task gradient; the aux target is stop-grad.
    Sows for the loss (read by mlm_train_step): 'mlm_ponder' (mean soft alpha,
    the escalation-cost term), 'mlm_aux' (Rao-Ballard: top predicts the NEXT
    window's pooled bottom error -> trains the top even when NOT escalated),
    'mlm_esc' (mean hard escalation rate, logged).
    Requires bidirectional=False, batchnorm=False (causal, no future leak), and
    sequence length L divisible by `mlm_stride`.
    """
    ssm: nn.Module
    d_output: int          # vocab size
    d_model: int
    n_layers: int          # BOTTOM layers
    padded: bool = False
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    mode: str = ""
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    glu_rank: int = 0
    glu_structure: str = "dense"
    # -- Mambino-LM knobs --
    mlm_stride: int = 4            # s: top ticks once per s tokens (slow ticking = FIX #1)
    mlm_top_layers: int = 2       # TOP encoder depth (small, the deliberate level)
    mlm_alpha_thresh: float = 0.0 # theta init (escalation threshold on z0)
    mlm_kappa_init: float = 4.0   # gate sharpness init
    mlm_beta_init: float = 1.0    # nudge magnitude init
    mlm_gate_ema: float = 0.9     # trailing-surprise EMA decay
    alpha_override: float = -99.0 # <-90: use training flag (soft/hard); else force this constant

    def setup(self):
        common = dict(ssm=self.ssm, d_model=self.d_model, activation=self.activation,
                      dropout=self.dropout, training=self.training, prenorm=self.prenorm,
                      batchnorm=self.batchnorm, bn_momentum=self.bn_momentum,
                      step_rescale=self.step_rescale, glu_rank=self.glu_rank,
                      glu_structure=self.glu_structure)
        self.bottom = StackedEncoderModel(n_layers=self.n_layers, **common)
        self.bottom_decoder = nn.Dense(self.d_output)
        self.top = StackedEncoderModel(n_layers=self.mlm_top_layers, **common)
        self.nudge_decoder = nn.Dense(self.d_output)
        self.aux_head = nn.Dense(1)                       # Rao-Ballard: predict pooled bottom error
        self.mlm_kappa = self.param("mlm_kappa", lambda r, s: np.full(s, self.mlm_kappa_init), (1,))
        self.mlm_theta = self.param("mlm_theta", lambda r, s: np.full(s, self.mlm_alpha_thresh), (1,))
        self.mlm_beta = self.param("mlm_beta", lambda r, s: np.full(s, self.mlm_beta_init), (1,))

    def _trailing_z(self, e):
        """e: (L-1,) bottom errors for positions 0..L-2.  Returns z0: (L,) where
        z0[t] = bias-corrected EMA of e[0..t-1] (strictly trailing; z0[0]=0)."""
        a = self.mlm_gate_ema

        def step(carry, et):
            m, cnt = carry
            m = a * m + (1.0 - a) * et
            cnt = cnt + 1.0
            mu = m / (1.0 - a ** cnt)                      # bias-corrected (Adam-style)
            return (m, cnt), mu

        _, M = jax.lax.scan(step, (0.0, 0.0), e)           # M[k] = EMA(e[0..k]), length L-1
        return np.concatenate([np.zeros((1,)), M], axis=0) # z0[t]=M[t-1]; length L

    def __call__(self, x, integration_timesteps):
        L = x.shape[0]
        s = self.mlm_stride
        W = L // s                                          # #windows (caller ensures L % s == 0)

        # ---- bottom (fast, per-token) ----
        F0 = self.bottom(x, integration_timesteps)          # (L, H)
        p0 = self.bottom_decoder(F0)                         # (L, V) pre-nudge bottom logits
        lp0 = nn.log_softmax(p0, axis=-1)

        # ---- realized trailing surprise z0 (causal; excludes current token) ----
        tgt = np.argmax(x[1:], axis=-1)                     # (L-1,) realized next tokens (pos 0..L-2)
        e = -lp0[np.arange(L - 1), tgt]                     # (L-1,) bottom CE errors
        z0 = self._trailing_z(jax.lax.stop_gradient(e))     # (L,) gate DECISION detaches surprise

        # ---- escalation gate: soft (train) / hard (eval) / forced (tests) ----
        soft_a = jax.nn.sigmoid(self.mlm_kappa[0] * (z0 - self.mlm_theta[0]))
        hard_a = (z0 > self.mlm_theta[0]).astype(np.float32)
        if self.alpha_override > -90.0:
            alpha = np.full((L,), np.float32(self.alpha_override))
        else:
            alpha = soft_a if self.training else hard_a

        # ---- top (slow tick over pooled bottom features), causal one-window shift ----
        Fpool = np.mean(F0.reshape(W, s, -1), axis=1)       # (W, H) window-mean of bottom features
        Gtop = self.top(Fpool, np.ones((W,)))               # (W, H) causal SSM over subsampled stream
        nudge_win = self.nudge_decoder(Gtop)                # (W, V)
        # window w reads the top's summary through window w-1 (all tokens strictly earlier -> causal)
        nudge_win_sh = np.concatenate([np.zeros((1, self.d_output)), nudge_win[:-1]], axis=0)  # (W,V)
        nudge = np.repeat(nudge_win_sh, s, axis=0)          # (L, V)

        logits = p0 + alpha[:, None] * self.mlm_beta[0] * nudge
        log_probs = nn.log_softmax(logits, axis=-1)

        # ---- Rao-Ballard aux: top predicts NEXT window's pooled bottom error ----
        e_full = np.concatenate([e, np.zeros((1,))], axis=0)          # (L,) pad last (unused as target below)
        e_pool = np.mean(e_full.reshape(W, s), axis=1)                # (W,) pooled bottom error per window
        aux_pred = self.aux_head(Gtop)[:, 0]                          # (W,) top's prediction
        aux_target = jax.lax.stop_gradient(e_pool)
        aux_loss = np.mean((aux_pred[:-1] - aux_target[1:]) ** 2)     # predict NEXT window (w -> w+1)

        # ---- sow terms for the loss / logging ----
        self.sow("intermediates", "mlm_aux", aux_loss)
        self.sow("intermediates", "mlm_ponder", np.mean(soft_a))
        self.sow("intermediates", "mlm_esc", np.mean(hard_a))
        return log_probs


BatchMambinoLMModel = nn.vmap(
    MambinoLMModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None, "intermediates": 0},
    split_rngs={"params": False, "dropout": True, "noise": True}, axis_name='batch')


# For Document matching task (e.g. AAN)
class RetrievalDecoder(nn.Module):
    """
    Defines the decoder to be used for document matching tasks,
    e.g. the AAN task. This is defined as in the S4 paper where we apply
    an MLP to a set of 4 features. The features are computed as described in
    Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
    Args:
        d_output    (int32):    the output dimension, i.e. the number of classes
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                    we usually refer to this size as H
    """
    d_model: int
    d_output: int

    def setup(self):
        """
        Initializes 2 dense layers to be used for the MLP.
        """
        self.layer1 = nn.Dense(self.d_model)
        self.layer2 = nn.Dense(self.d_output)

    def __call__(self, x):
        """
        Computes the input to be used for the softmax function given a set of
        4 features. Note this function operates directly on the batch size.
        Args:
             x (float32): features (bsz, 4*d_model)
        Returns:
            output (float32): (bsz, d_output)
        """
        x = self.layer1(x)
        x = nn.gelu(x)
        return self.layer2(x)


class RetrievalModel(nn.Module):
    """ S5 Retrieval classification model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, constructing 4 features which are fed into a MLP,
    and a softmax operation. Note that unlike the standard classification model above,
    the apply function of this model operates directly on the batch of data (instead of calling
    vmap on this model).
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
    """
    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    padded: bool
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes the S5 stacked encoder and the retrieval decoder. Note that here we
        vmap over the stacked encoder model to work well with the retrieval decoder that
        operates directly on the batch.
        """
        BatchEncoderModel = nn.vmap(
            StackedEncoderModel,
            in_axes=(0, 0),
            out_axes=0,
            variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None, "intermediates": 0},
            split_rngs={"params": False, "dropout": True}, axis_name='batch'
        )

        self.encoder = BatchEncoderModel(
                            ssm=self.ssm,
                            d_model=self.d_model,
                            n_layers=self.n_layers,
                            activation=self.activation,
                            dropout=self.dropout,
                            training=self.training,
                            prenorm=self.prenorm,
                            batchnorm=self.batchnorm,
                            bn_momentum=self.bn_momentum,
                            step_rescale=self.step_rescale,
                                        )
        BatchRetrievalDecoder = nn.vmap(
            RetrievalDecoder,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )

        self.decoder = BatchRetrievalDecoder(
                                d_model=self.d_model,
                                d_output=self.d_output
                                          )

    def __call__(self, input, integration_timesteps):  # input is a tuple of x and lengths
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence. The encoded features are constructed as in
        Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
        Args:
             input (float32, int32): tuple of input sequence and prepadded sequence lengths
                input sequence is of shape (2*bsz, L, d_input) (includes both documents) and
                lengths is (2*bsz,)
        Returns:
            output (float32): (d_output)
        """
        x, lengths = input  # x is 2*bsz*seq_len*in_dim, lengths is: (2*bsz,)
        x = self.encoder(x, integration_timesteps)  # The output is: 2*bszxseq_lenxd_model
        outs = batch_masked_meanpool(x, lengths)  # Avg non-padded values: 2*bszxd_model
        outs0, outs1 = np.split(outs, 2)  # each encoded_i is bszxd_model
        features = np.concatenate([outs0, outs1, outs0-outs1, outs0*outs1], axis=-1)  # bszx4*d_model
        out = self.decoder(features)
        return nn.log_softmax(out, axis=-1)
