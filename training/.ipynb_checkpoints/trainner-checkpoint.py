import multiprocessing
# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    WhisperFeatureExtractor, WhisperForConditionalGeneration,
    WhisperProcessor, WhisperTokenizer
)
from huggingface_hub import HfFolder

from training.DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding
from training.MetricsEval import MetricsEval


from config import model_dir, HF_API_KEY, BASE_MODEL

# training constants
TRAIN_BATCH_SIZE = 1
GRAIDENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
MAX_STEPS = 4000
EVAL_BATCH_SIZE = 1
SAVE_STEPS = 1000
EVAL_STEPS = 1000
LOGGING_STEPS = 25


class WhisperASR:
    """Whisper Model for Automatic Speech Recognition (ASR) using Hugging Face's Transformers library."""

    def __init__(self, model_name="openai/whisper-small", dataset_name="mozilla-foundation/common_voice_13_0", existing_model=False, language="Greek", language_code="el", save_to_hf=False, output_dir="./models/whisper", ref_key="text"):
        """
        Initialize the model and load the data. 
        The default config is the small model trained on the Common Voice dataset for Greek.

        Args:
            model_name (str): The model name from Hugging Face or custom path
            If 'existing_model' is True, this should be the path to the pre-trained model. Ex: "openai/whisper-small"

            existing_model (bool): Flag to indicate whether to load an existing model from the specified 
            'model_name' path. If False, a new model is initialized

            language (str): The language of the model. Ex: "Greek"
            language_code (str): The dataset configuration/language code. Ex: "el"
            output_dir (str): The output directory of the model to save to
            save_to_hf (bool): Whether to push to Hugging Face Repo
            ref_key (str): The key to the reference data in the dataset
        """
        # setting up to save to hugging face repo
        self.save_to_hf = save_to_hf
        if save_to_hf:
            HfFolder.save_token(HF_API_KEY)  # token to save to HF

        self.dataset_name = dataset_name
        self.ref_key = ref_key

        # initialize model and tokenizer
        self.model_name = model_name
        self.language = language
        self.language_code = language_code
        self.existing_model = existing_model

        self.train_split = "train"
        self.test_split = "test"

        # initalize feature extractor, tokenizer and processor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            self.model_name, language=language, task="transcribe")
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, language=language, task="transcribe")

        # load correct model
        if existing_model:
            print(
                f"[INFO] Loading {self.model_name} model from existing model...")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        else:
            print(
                f"[INFO] Loading {self.model_name} from hugging face library...")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name)

        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        
        # Set correct language code for Hugging Face Hub
        if hasattr(self.model.config, 'language'):
            self.model.config.language = self.language_code  # Use "el" instead of "Greek"
        
        # Configure model for memory optimization
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False
        
        # Disable gradient checkpointing completely to prevent gradient issues
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        
        # Additional memory optimizations
        if hasattr(self.model.config, 'torch_dtype'):
            self.model.config.torch_dtype = "float16"
        
        # Move model to GPU with memory optimization
        import torch
        if torch.cuda.is_available():
            # Ensure CUDA is properly initialized
            torch.cuda.init()
            self.model = self.model.cuda()
            torch.cuda.empty_cache()  # Clear cache after moving model

        # load data
        self.data = DatasetDict()
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            self.processor)
        self._load_data()
        self.OUTPUT_DIR = output_dir

    def _load_data(self):
        """Load the data from the Common Voice dataset and prepare it for training."""

        print(
            f"[INFO] Preparing {self.dataset_name} data for training phase...")

        # load data from Common Voice dataset
        self.data["train"] = load_dataset(
            self.dataset_name, split=self.train_split, token=HF_API_KEY)
        self.data["test"] = load_dataset(
            self.dataset_name, split=self.test_split, token=HF_API_KEY)

        print("[INFO] Structure of the loaded data:")
        print(self.data)

        print("[INFO] Sample entry from the training dataset: ")
        print(self.data["train"][0])

        # downsample audio data to 16kHz
        self.data = self.data.cast_column("audio", Audio(sampling_rate=16000))
        self.data = self.data.map(
            self._prepare_data, remove_columns=self.data.column_names["train"], num_proc=0)  # Disable multiprocessing completely

    def _prepare_data(self, batch):
        """Converts audio files to the model's input feature format and encodes the target texts.

        Args:
            batch (dict): A batch of audio and text data.
        """

        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch[self.ref_key]).input_ids
        return batch

    def train(self):
        """Train the model. Set the training arguments here and using Seq2SeqTrainer. 
        After training, save the model to the specified directory.
        """

        # metric evaluation for training
        eval_fn = MetricsEval(self.tokenizer)

        # configure training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.OUTPUT_DIR,
            per_device_train_batch_size=1,  # Minimal batch size for memory
            gradient_accumulation_steps=16,  # Increased to maintain effective batch size
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            gradient_checkpointing=False,  # Disabled to prevent gradient issues
            fp16=True,  # Enable fp16 to save memory
            eval_strategy="no",  # Disable evaluation to prevent gradient issues
            per_device_eval_batch_size=1,  # Minimal eval batch size
            predict_with_generate=False,  # Disable generation to prevent gradient issues
            generation_max_length=225,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            report_to=[],  # Disable tensorboard to save memory
            load_best_model_at_end=False,  # Disable to prevent gradient issues
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=self.save_to_hf,
            dataloader_drop_last=True,  # Drop last incomplete batch
            remove_unused_columns=True,  # Remove unused columns to save memory
            dataloader_num_workers=0,  # Disable multiprocessing for stability
            save_total_limit=1,  # Keep only 1 checkpoint
            dataloader_pin_memory=False,  # Disable pin memory to save RAM
            max_grad_norm=1.0,  # Gradient clipping
            optim="adamw_torch",  # Use AdamW optimizer
            lr_scheduler_type="linear",  # Linear learning rate scheduler
            warmup_ratio=0.1,  # Warmup ratio
            dataloader_prefetch_factor=None,  # Disable prefetching
            ddp_find_unused_parameters=False,  # Disable unused parameter detection
        )

        # initialize trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            data_collator=self.data_collator,
            compute_metrics=eval_fn.compute,
            processing_class=self.processor,  # Use processing_class instead of tokenizer
        )

        self.processor.save_pretrained(training_args.output_dir)

        # start training
        print("[INFO] Starting training...: ")
        trainer.train()

        # training finished and save model to model directory
        print(f"[INFO] Training finished and model saved to {self.OUTPUT_DIR}")

        if self.save_to_hf:
            kwargs = {
                "language": f"{self.language}",
                "model_name": f"Whisper - {self.language} Model",
                "finetuned_from": f"{BASE_MODEL}",
                "tasks": "automatic-speech-recognition",
            }

            trainer.push_to_hub(**kwargs)
            print(f"[INFO] Model saved to Hugging Face Hub")
