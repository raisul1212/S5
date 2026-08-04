import argparse
from s5.utils.util import str2bool
from s5.train import train
from s5.dataloading import Datasets

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--USE_WANDB", type=str2bool, default=False,
						help="log with wandb?")
	parser.add_argument("--wandb_project", type=str, default=None,
						help="wandb project name")
	parser.add_argument("--wandb_entity", type=str, default=None,
						help="wandb entity name, e.g. username")
	parser.add_argument("--dir_name", type=str, default='./cache_dir',
						help="name of directory where data is cached")
	parser.add_argument("--dataset", type=str, choices=Datasets.keys(),
						default='mnist-classification',
						help="dataset name")
	parser.add_argument("--val_split", type=float, default=0.0,
						help="Fraction of TRAIN held out as a validation split, for "
							 "datasets that ship none. 0.0 (default) preserves the "
							 "released behaviour exactly: the LRA convention assigns "
							 "the test set to the validation role, so the reported "
							 "checkpoint is selected on test. Set >0 to select "
							 "checkpoints on genuinely held-out data instead. "
							 "Currently honoured by imdb-classification only; other "
							 "loaders ignore it and are byte-identical either way.")

	# Model Parameters
	parser.add_argument("--n_layers", type=int, default=6,
						help="Number of layers in the network")
	parser.add_argument("--d_model", type=int, default=128,
						help="Number of features, i.e. H, "
							 "dimension of layer inputs/outputs")
	parser.add_argument("--ssm_size_base", type=int, default=256,
						help="SSM Latent size, i.e. P")
	parser.add_argument("--blocks", type=int, default=8,
						help="How many blocks, J, to initialize with")
	parser.add_argument("--C_init", type=str, default="trunc_standard_normal",
						choices=["trunc_standard_normal", "lecun_normal", "complex_normal"],
						help="Options for initialization of C: \\"
							 "trunc_standard_normal: sample from trunc. std. normal then multiply by V \\ " \
							 "lecun_normal sample from lecun normal, then multiply by V\\ " \
							 "complex_normal: sample directly from complex standard normal")
	parser.add_argument("--discretization", type=str, default="zoh", choices=["zoh", "bilinear"])
	parser.add_argument("--mode", type=str, default="pool", choices=["pool", "last"],
						help="options: (for classification tasks) \\" \
							 " pool: mean pooling \\" \
							 "last: take last element")
	parser.add_argument("--activation_fn", default="half_glu1", type=str,
						choices=["full_glu", "half_glu1", "half_glu2", "gelu"])
	parser.add_argument("--conj_sym", type=str2bool, default=True,
						help="whether to enforce conjugate symmetry")
	parser.add_argument("--clip_eigs", type=str2bool, default=False,
						help="whether to enforce the left-half plane condition")
	parser.add_argument("--bidirectional", type=str2bool, default=False,
						help="whether to use bidirectional model")
	parser.add_argument("--dt_min", type=float, default=0.001,
						help="min value to sample initial timescale params from")
	parser.add_argument("--dt_max", type=float, default=0.1,
						help="max value to sample initial timescale params from")

	# Optimization Parameters
	parser.add_argument("--prenorm", type=str2bool, default=True,
						help="True: use prenorm, False: use postnorm")
	parser.add_argument("--batchnorm", type=str2bool, default=True,
						help="True: use batchnorm, False: use layernorm")
	parser.add_argument("--bn_momentum", type=float, default=0.95,
						help="batchnorm momentum")
	parser.add_argument("--bsz", type=int, default=64,
						help="batch size")
	parser.add_argument("--epochs", type=int, default=100,
						help="max number of epochs")
	parser.add_argument("--early_stop_patience", type=int, default=1000,
						help="number of epochs to continue training when val loss plateaus")
	parser.add_argument("--ssm_lr_base", type=float, default=1e-3,
						help="initial ssm learning rate")
	parser.add_argument("--lr_factor", type=float, default=1,
						help="global learning rate = lr_factor*ssm_lr_base")
	parser.add_argument("--dt_global", type=str2bool, default=False,
						help="Treat timescale parameter as global parameter or SSM parameter")
	parser.add_argument("--lr_min", type=float, default=0,
						help="minimum learning rate")
	parser.add_argument("--cosine_anneal", type=str2bool, default=True,
						help="whether to use cosine annealing schedule")
	parser.add_argument("--warmup_end", type=int, default=1,
						help="epoch to end linear warmup")
	parser.add_argument("--lr_patience", type=int, default=1000000,
						help="patience before decaying learning rate for lr_decay_on_val_plateau")
	parser.add_argument("--reduce_factor", type=float, default=1.0,
						help="factor to decay learning rate for lr_decay_on_val_plateau")
	parser.add_argument("--p_dropout", type=float, default=0.0,
						help="probability of dropout")
	parser.add_argument("--weight_decay", type=float, default=0.05,
						help="weight decay value")
	parser.add_argument("--opt_config", type=str, default="standard", choices=['standard',
																			   'BandCdecay',
																			   'BfastandCdecay',
																			   'noBCdecay'],
						help="Opt configurations: \\ " \
			   "standard:       no weight decay on B (ssm lr), weight decay on C (global lr) \\" \
	  	       "BandCdecay:     weight decay on B (ssm lr), weight decay on C (global lr) \\" \
	  	       "BfastandCdecay: weight decay on B (global lr), weight decay on C (global lr) \\" \
	  	       "noBCdecay:      no weight decay on B (ssm lr), no weight decay on C (ssm lr) \\")
	parser.add_argument("--jax_seed", type=int, default=1919,
						help="seed randomness")

	# ── MambinoSSM extension (drop-in replacement for S5SSM) ──
	parser.add_argument("--use_mambino_ssm", type=str2bool, default=False,
						help="If True, replace S5SSM with MambinoSSM "
							 "(adds proprioceptive predictor branch + "
							 "W_eps additive PC path).  All other S5 "
							 "machinery is unchanged.  Default False = "
							 "vanilla S5.")
	parser.add_argument("--lambda_pc", type=float, default=0.0,
						help="Weight on MambinoSSM's intrinsic loss "
							 "(L_int = mean ||eps||^2 summed across "
							 "blocks).  Added to task CE: "
							 "total = task_CE + lambda_pc * L_int.  "
							 "Default 0.0 = predictor branch is "
							 "architecturally active but no separate "
							 "prediction-quality gradient.  Matches "
							 "the 0.4965 Mambino hero recipe.  "
							 "Ignored when --use_mambino_ssm=False.")
	parser.add_argument("--ckpt_dir", type=str, default="",
						help="Directory for best.pkl + final.pkl "
							 "checkpoints.  Empty = do not save "
							 "(preserves original S5 behavior).  "
							 "Recommended: set to a per-run output "
							 "directory so reload is possible.")
	parser.add_argument("--glu_rank", type=int, default=0,
						help="If > 0 and activation is half_glu*, "
							 "factorize the out2 Dense(H, H) into "
							 "Dense(H, r) @ Dense(r, H) with r=glu_rank. "
							 "Reduces gate params from ~2H^2 to 2Hr. "
							 "Used for iso-param comparisons: shrink the "
							 "gate to redirect params to bigger SSM state "
							 "(via --ssm_size_base).  Default 0 = full-rank.")
	parser.add_argument("--glu_structure", type=str, default="dense",
						choices=["dense", "monarch", "blockdiag"],
						help="v2 structured gate (half_glu2 only). 'dense' = v1 "
							 "path (byte-identical). 'monarch' = R-head Monarch(b,m) "
							 "operator (size via --glu_monarch_heads). 'blockdiag' = "
							 "B diagonal blocks (size via --glu_blockdiag_blocks). "
							 "For monarch/blockdiag, --glu_rank only needs to be >0 "
							 "as the gate-on trigger; the structured op's size is set "
							 "by its own knob.")
	parser.add_argument("--glu_monarch_heads", type=int, default=3,
						help="Monarch heads R (params = R*H*(b+m)+H). "
							 "At H=128 (b,m)=(8,16): R=3 -> 9,216 params.")
	parser.add_argument("--glu_monarch_b", type=int, default=0,
						help="Force Monarch factor b (m=H/b). 0=auto (8,16 at H=128). "
							 "16 -> (16,8): same params, halves inner-einsum batch (lower latency).")
	parser.add_argument("--glu_monarch_residual_rank", type=int, default=0,
						help="Add a rank-r' dense residual to the Monarch gate (+2*H*r' "
							 "params). r'=4 at H=128 -> +1,024/layer = iso-param with dense r=40.")
	parser.add_argument("--glu_blockdiag_blocks", type=int, default=2,
						help="block-diagonal block count B (params = H^2/B + H). "
							 "At H=128: B=2 -> 8,192 params.")
	parser.add_argument("--bidir_predictor", type=str2bool, default=False,
						help="If True, MambinoSSM's predictor scan runs "
							 "bidirectionally (mirrors main scan), with "
							 "an additional C_s2 readout for the backward "
							 "direction.  x_hat(t) combines forward "
							 "(causal, s_fwd(t-1)) + backward "
							 "(anti-causal, s_bwd(t+1)) predictions. "
							 "Adds ~2K params per layer (16K total for "
							 "L=8).  Default False = causal forward-only "
							 "predictor (streaming-inference-compatible). "
							 "Ignored when --use_mambino_ssm=False.")
	parser.add_argument("--surprise_gate", type=str2bool, default=False,
						help="v2 Cluster-A signed adaptive gate on eps->W_eps inside MambinoSSM: g=tanh(kappa*z+bias), z=running-EMA-normalized ||eps||. Off (default) => byte-identical kill-switch; +2 scalars/layer.")
	parser.add_argument("--gate_alpha", type=float, default=0.9,
						help="EMA decay of the surprise-gate running normalizer.")
	parser.add_argument("--gate_range", type=str, default="signed",
						choices=["signed", "unsigned"],
						help="signed=tanh[-1,1] (push/hold/pop); unsigned=sigmoid[0,1].")
	parser.add_argument("--gate_kappa_init", type=float, default=0.0,
						help="Init for gate sensitivity kappa (0 => g starts flat).")
	parser.add_argument("--gate_bias_init", type=float, default=2.0,
						help="Init for gate bias (+2 => g~0.96 = ~v1 write, livelier kappa grad).")
	parser.add_argument("--gate_detach", type=str2bool, default=False,
						help="Also stop-gradient eps in the W_eps write (calibration fix); gate DECISION always detaches.")

	# ── v2 Cluster-B: surprise-gated fast weight (role 2 = inference-time learning) ──
	parser.add_argument("--fast_weight", type=str2bool, default=False,
						help="v2 Cluster-B fast weight: M_t = gamma*M_{t-1} + (1-gamma)*s*(v k^T), "
							 "o_t = M_{t-1} q_t, added to the block output. M is per-sequence STATE, "
							 "not parameters. Off (default) => byte-identical kill-switch.")
	parser.add_argument("--fw_dim", type=int, default=8,
						help="d: q/k/v dim; the fast weight M is d x d. shared d=8 => +17.0%% params.")
	parser.add_argument("--fw_proj", type=str, default="shared",
						choices=["shared", "separate"],
						help="shared = 1 H->d down-proj + 3 dxd mixers (2Hd+3d^2); separate = 3 H->d (4Hd).")
	parser.add_argument("--fw_rule", type=str, default="hebb", choices=["hebb", "delta"],
						help="hebb = B0 outer-product write; delta = B1 error-correcting write (NOT yet implemented).")
	parser.add_argument("--fw_kq_source", type=str, default="x", choices=["x", "eps"],
						help="Address space for k and q. Default x keeps write-address and query in one anchored space.")
	parser.add_argument("--fw_v_source", type=str, default="eps", choices=["x", "eps"],
						help="Content written into M. Default eps = the SIGNED first moment (theory-pure); "
							 "x is the ablation isolating whether surprise-as-content matters beyond the gate.")
	parser.add_argument("--fw_impl", type=str, default="chunk", choices=["seq", "scan", "chunk"],
						help="seq = sequential ground truth (slow); scan = associative_scan, O(L*d^2) memory; "
							 "chunk = chunked, O(L*d + n*d^2). All three are verified equivalent.")
	parser.add_argument("--fw_chunk", type=int, default=64,
						help="Chunk length C for --fw_impl=chunk.")
	parser.add_argument("--fw_gate_mode", type=str, default="surprise",
						choices=["surprise", "const", "off"],
						help="surprise = sigmoid(kappa*z+bias) (the hypothesis); const = kappa pinned to 0, "
							 "bias learned (THE iso-param ablation: same param count, same free write rate, "
							 "differs only in z-dependence); off = ungated, s==1.")
	parser.add_argument("--fw_kappa_init", type=float, default=0.0,
						help="Init for the fast-weight gate sensitivity kappa.")
	parser.add_argument("--fw_bias_init", type=float, default=0.0,
						help="Init for the fast-weight gate bias (0 => sigmoid=0.5, half-strength write).")
	parser.add_argument("--fw_gamma_init", type=float, default=0.95,
						help="Fast-weight decay gamma (memory length ~ 1/(1-gamma)). Stored as a logit when trainable.")
	parser.add_argument("--fw_gamma_trainable", type=str2bool, default=True,
						help="False => gamma frozen at --fw_gamma_init EXACTLY. Use with --fw_gamma_init=0 "
							 "for the no-memory control arm.")
	parser.add_argument("--fw_norm_qkv", type=str2bool, default=True,
						help="L2-normalize q, k AND v. Normalizing v matters: otherwise ||v|| ~ ||eps|| puts "
							 "surprise into the write twice and blunts the gated-vs-const ablation.")
	parser.add_argument("--fw_out_init", type=str, default="zeros", choices=["zeros", "lecun"],
						help="zeros => the branch is an exact no-op at step 0, so training STARTS at the baseline.")
	parser.add_argument("--fw_read", type=str, default="exclusive",
						choices=["exclusive", "inclusive"],
						help="exclusive = o_t reads M_{t-1} (read before write). inclusive lets the layer learn "
							 "W_q~W_k and degenerate into a gated INSTANTANEOUS path that uses no memory.")
	parser.add_argument("--chip_eval_sigmas", type=str, default="",
						help="Comma-separated analog noise sigmas to sweep at "
							 "end of training (e.g. '0,0.01,0.02,0.05,0.08'). "
							 "Empty = no chip sweep.")
	parser.add_argument("--chip_eval_bits", type=str, default="",
						help="Comma-separated ADC bit depths to sweep at "
							 "end of training (e.g. '0,4,5,6,7,8').  0 = no "
							 "quantization.  Empty = no chip sweep.")

	train(parser.parse_args())
