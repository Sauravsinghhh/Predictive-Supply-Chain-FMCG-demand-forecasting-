"""
Custom Exception Classes for FreshMind.
Defines meaningful exceptions for data loading, validation, feature engineering, and inference.
"""

class FreshMindError(Exception):
    """Base exception class for all FreshMind errors."""
    pass

class MissingDatasetError(FreshMindError):
    """Exception raised when required M5 dataset files are missing."""
    def __init__(self, file_path: str, message: str = "Required dataset file not found"):
        self.file_path = file_path
        self.message = f"{message}: {file_path}"
        super().__init__(self.message)

class InvalidStoreError(FreshMindError):
    """Exception raised when an invalid store ID is requested."""
    def __init__(self, store_id: str, message: str = "Invalid or non-existent store ID"):
        self.store_id = store_id
        self.message = f"{message}: '{store_id}'"
        super().__init__(self.message)

class InvalidSKUError(FreshMindError):
    """Exception raised when an invalid SKU/item ID is requested."""
    def __init__(self, item_id: str, message: str = "Invalid or non-existent SKU/item ID"):
        self.item_id = item_id
        self.message = f"{message}: '{item_id}'"
        super().__init__(self.message)

class EmptyDataFrameError(FreshMindError):
    """Exception raised when an operation receives or produces an empty DataFrame."""
    def __init__(self, dataset_name: str, message: str = "DataFrame is empty"):
        self.dataset_name = dataset_name
        self.message = f"{message}: {dataset_name}"
        super().__init__(self.message)

class DataValidationError(FreshMindError):
    """Exception raised when loaded datasets fail schema or integrity validations."""
    def __init__(self, detail: str, message: str = "Data validation check failed"):
        self.detail = detail
        self.message = f"{message}: {detail}"
        super().__init__(self.message)

class ConfigurationError(FreshMindError):
    """Exception raised when project configuration is missing, invalid, or corrupted."""
    pass
