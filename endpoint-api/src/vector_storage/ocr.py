# --- Standard library imports ---
import logging
from pathlib import Path
from typing import Optional, Union

# --- Third-party imports ---
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles image preprocessing and text extraction using Tesseract OCR"""

    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize OCR engine with optional Tesseract path"""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Default preprocessing pipeline (can be extended if needed)
        self.preprocessing_steps = [
            self.convert_to_grayscale # Basic grayscale conversion
            ]
        logger.info("OCRProcessor initialized with default preprocessing pipeline")

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale with comprehensive color handling"""
        if len(image.shape) == 3:
            # Handle all possible color formats in one step
            if image.shape[2] == 4:  # RGBA
                logger.debug("Converting RGBA to grayscale")
                logger.debug("Converting RGB to grayscale")
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 3:  # RGB
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                raise ValueError(f"Unsupported color channels: {image.shape[2]}")
        return image  # Already grayscale
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps to enhance OCR accuracy"""
        processed = image.copy()
        for step in self.preprocessing_steps:
            processed = step(processed)
        return processed
    
    def extract_text(self, image: Union[str, Path, np.ndarray], 
                    lang: str = 'eng', 
                    config: str = '--psm 6') -> str:
        """
        Extract text from image with preprocessing
        
        Args:
            image: Path/Path object/numpy array
            lang: ISO 639-2 language code (e.g., 'eng', 'spa')
            config: Tesseract config flags (--psm 6 = assume single uniform text block)        
        """
        logger.info(f"Starting text extraction (lang: {lang}, config: {config})")
        try:
            # Handle different input types
            if isinstance(image, (str, Path)):
                logger.info(f"Reading image from path: {image}")
                img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Could not read image from {image}")
            else:
                logger.debug("Processing in-memory image")
                img = image.copy()
            
            processed = self.preprocess_image(img)
            custom_config = f'-l {lang} {config}'

            logger.debug("Running Tesseract OCR")
            return pytesseract.image_to_string(processed, config=custom_config).strip()
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def extract_text_with_data(self, image: Union[str, Path, np.ndarray],
                               lang: str = 'eng',
                               config: str = '--psm 6',
                               min_confidence: float = 40.0,
                               return_image_with_boxes: bool = False) -> dict:
        """
        Extract text with bounding box and confidence information.
        
        Args:
            image: Path or image array
            lang: Language for OCR
            config: Tesseract config
            min_confidence: Confidence threshold
            return_image_with_boxes: If True, returns image with rectangles around confident text
    
        Returns:
            dict: 'text': extracted text, 'data': list of word metadata, 'image': annotated image (optional)
        """
        logger.info(f"Starting detailed extraction (conf: {min_confidence}%)")
        try:
            if isinstance(image, (str, Path)):
                img = cv2.imread(str(image))
                if img is None:
                    raise ValueError(f"Could not read image from {image}")
            else:
                logger.debug("Processing in-memory image")
                img = image.copy()
    
            original_img = img.copy() if return_image_with_boxes else None
            processed = self.preprocess_image(img)
            custom_config = f'-l {lang} {config}'
    
            logger.debug("Running Tesseract with data output") 
            data = pytesseract.image_to_data(
                processed,
                output_type=Output.DICT,
                config=custom_config
            )
    
            filtered_text = []
            filtered_data = []
    
            for i in range(len(data['text'])):
                try:
                    confidence = float(data['conf'][i])
                except ValueError:
                    continue  # Skip empty or invalid entries
                if confidence > min_confidence:
                    word_data = {
                        'text': data['text'][i],
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': confidence
                    }
                    filtered_text.append(data['text'][i])
                    filtered_data.append(word_data)
    
                    if return_image_with_boxes and original_img is not None:
                        x, y, w, h = word_data['left'], word_data['top'], word_data['width'], word_data['height']
                        cv2.rectangle(original_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
            result = {
                'text': ' '.join(filtered_text),
                'data': filtered_data
            }
            if return_image_with_boxes:
                result['image'] = original_img
            return result
    
        except Exception as e:
            logger.error(f"Detailed OCR failed: {e}", exc_info=True)
            raise