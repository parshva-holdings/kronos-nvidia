"""Indian-market fine-tune config — drop-in replacement for upstream/finetune/config.py.

Activated by `scripts/07_finetune_max.sh` which copies this file over the upstream
config at run time. Supports both daily (default) and intraday training.

Tuning knobs you'll most likely change:
  - lookback_window / predict_window  (window must fit max_context = 512)
  - batch_size                        (per-GPU; 50 fits easily on H100-80, 24 on A100-40)
  - epochs                            (30 is upstream default; 15 often plenty for fine-tune)
  - n_train_iter                      (samples per epoch; 100K = ~16h on 8x H100 for predictor)
  - tokenizer_learning_rate / predictor_learning_rate
"""
import os


class Config:
    def __init__(self):
        # ----------------------------------------------------------
        # Data — we bypass Qlib entirely. The pickles produced by
        # data/build_indian_corpus.py already match the dict[sym]->DataFrame
        # schema that upstream/finetune/dataset.py expects.
        # ----------------------------------------------------------
        self.qlib_data_path  = "<unused — bypassed>"
        self.instrument      = "nifty500"

        self.dataset_begin_time = "1996-07-01"
        self.dataset_end_time   = "2026-12-31"

        # Window sizing. Kronos-base has 512-token context; window = 400 + 100 + 1 = 501
        # leaves headroom for the BOS / time-feature tokens upstream may add.
        self.lookback_window = 400
        self.predict_window  = 100
        self.max_context     = 512

        self.feature_list      = ['open', 'high', 'low', 'close', 'vol', 'amt']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # ----------------------------------------------------------
        # Splits — match what build_indian_corpus.py emits
        # ----------------------------------------------------------
        self.train_time_range = ["1996-07-01", "2022-12-31"]
        self.val_time_range   = ["2022-09-01", "2024-12-31"]
        self.test_time_range  = ["2024-10-01", "2026-12-31"]
        self.backtest_time_range = ["2025-01-01", "2026-12-31"]

        self.dataset_path = os.environ.get(
            "KRONOS_DATASET_PATH",
            os.path.expanduser("~/kronos_data/processed_datasets"),
        )

        # ----------------------------------------------------------
        # Training hyperparameters
        # ----------------------------------------------------------
        self.clip   = 5.0
        self.epochs = int(os.environ.get("KRONOS_EPOCHS", 30))

        # batch_size is per-GPU. Override via env for smaller cards.
        self.batch_size = int(os.environ.get("KRONOS_BATCH_SIZE", 50))
        self.log_interval = 50

        # Samples per epoch (a virtual epoch since the real corpus is huge).
        self.n_train_iter = int(os.environ.get("KRONOS_N_TRAIN_ITER", 100_000))
        self.n_val_iter   = int(os.environ.get("KRONOS_N_VAL_ITER",   8_000))

        self.tokenizer_learning_rate  = float(os.environ.get("KRONOS_TOKENIZER_LR", 2e-4))
        self.predictor_learning_rate  = float(os.environ.get("KRONOS_PREDICTOR_LR", 4e-5))

        self.accumulation_steps = int(os.environ.get("KRONOS_GRAD_ACCUM", 1))

        # AdamW
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        self.seed = int(os.environ.get("KRONOS_SEED", 100))

        # ----------------------------------------------------------
        # Comet ML — opt-in. Off by default to keep secrets clean.
        # ----------------------------------------------------------
        self.use_comet = os.environ.get("COMET_API_KEY") not in (None, "")
        self.comet_config = {
            "api_key":    os.environ.get("COMET_API_KEY", ""),
            "project_name": os.environ.get("COMET_PROJECT", "kronos-nifty-finetune"),
            "workspace":  os.environ.get("COMET_WORKSPACE", ""),
        }
        self.comet_tag  = "nifty-max"
        self.comet_name = "nifty-max"

        # ----------------------------------------------------------
        # Save paths
        # ----------------------------------------------------------
        self.save_path = os.environ.get(
            "KRONOS_SAVE_PATH", os.path.expanduser("~/kronos_data/models")
        )
        self.tokenizer_save_folder_name = "kronos_tokenizer_nifty"
        self.predictor_save_folder_name = "kronos_predictor_nifty"
        self.backtest_save_folder_name  = "kronos_backtest_nifty"
        self.backtest_result_path       = os.path.expanduser("~/kronos_data/backtest")

        # ----------------------------------------------------------
        # Pretrained starting points
        # ----------------------------------------------------------
        self.pretrained_tokenizer_path = os.environ.get(
            "KRONOS_PRETRAIN_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base"
        )
        self.pretrained_predictor_path = os.environ.get(
            "KRONOS_PRETRAIN_MODEL", "NeoQuasar/Kronos-base"
        )

        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )

        # ----------------------------------------------------------
        # Backtest knobs (only used if you run upstream's qlib_test.py;
        # for NSE we provide a separate backtest that lives in this kit).
        # ----------------------------------------------------------
        self.backtest_n_symbol_hold = 50
        self.backtest_n_symbol_drop = 5
        self.backtest_hold_thresh = 5
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = "^NSEI"  # not used unless running with Qlib
