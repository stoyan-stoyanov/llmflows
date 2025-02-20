"""
This module contains the Image class, which is used to represent for LLM calls.
"""
import base64
import httpx
import os
from mimetypes import guess_type


class Image:
    """
    Represents an image for LLM calls. The image can be a URL or a base64 encoded string.

    Args:
        url (str): The URL of the image.
    """
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]

    IMAGE_TYPE_EXTENSION_MAP = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    def __init__(self, image: str):
        self.image = image
        self.url = None
        self.file_path = None
        self.base64 = None
        self.image_extension = None
        self.image_mime_type = None
        self._check_image_definition()

    def _check_image_definition(self) -> None:
        """
        Checks if image is an URL, a file, or a base64 encoded string.
        If image is an URL, it will be downloaded and converted to a base64 encoded string.
        If image is a file, it will be converted to a base64 encoded string.
        If image is a base64 encoded string, it will be converted to a base64 encoded string
        with the correct MIME type.

        Raises:
            ValueError: If image is not an URL, a file, or a base64 encoded string.
        """

        if not self._is_url() and not self._is_base64_encoded(self.image) and not self._is_file():
            raise ValueError("Image must be an URL or a base64 encoded string or a file.")
        if self._is_url():
            print("Image is an URL")
            self.url = self.image
            print("URL: ", self.url)    
            self.image_extension, self.image_mime_type = self._get_image_type_from_url(self.image)
            print("Image extension: ", self.image_extension)
            print("Image MIME type: ", self.image_mime_type)
            self.base64 = self._url_to_base64()
        elif self._is_file():
            print("Image is a file")
            self.file_path = self.image
            self.image_extension = self._get_image_type_from_file(self.file_path)
            self.image_mime_type = self.IMAGE_TYPE_EXTENSION_MAP.get(self.image_extension)
            self.base64 = self._file_to_base64()
        else:
            print("Image is a base64 encoded string")
            mime_type, extension = guess_type(self.image)
            if not mime_type or not extension:
                mime_type = self._get_image_mime_type(self.image)
                extension = self.IMAGE_TYPE_EXTENSION_MAP.get(mime_type)
                # self.base64 = self._add_mime_type_to_base64(mime_type, self.image)
            else:
                self.base64 = self.image
            self.image_extension = extension
            self.image_mime_type = mime_type

        if self.image_mime_type is None:
            raise ValueError("Failed to infer image MIME type.")

        if self.image_mime_type not in self.SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Image type {self.image_mime_type} is not supported.")
        
        if self.image_extension is None or self.image_extension == "":
            raise ValueError("Failed to infer image extension. Resulting extension is empty.")
        
        if self.image_extension is not None and self.image_extension not in self.IMAGE_TYPE_EXTENSION_MAP:
            raise ValueError(f"Image extension {self.image_extension} is not supported.")

        if self.base64 is None or self.base64 == "":
            raise ValueError("Failed to convert image to base64 encoded string. Resulting string is empty.")

        if not self._is_base64_encoded(self.base64):
            raise ValueError(f"Failed to convert image to base64 encoded string. Resulting string {self.base64} is not base64 encoded.")

    def _is_url(self) -> bool:
        """
        Checks if image is an URL.

        Returns:
            bool: True if image is an URL, False otherwise.
        """
        return self.image.startswith("http")

    def _is_file(self) -> bool:
        """
        Checks if image is a file.

        Returns:
            bool: True if image is a file, False otherwise.
        """
        return os.path.isfile(self.image)

    def _is_base64_encoded(self, encoded_string: str) -> bool:
        """
        Checks if image is a base64 encoded string.

        Returns:
            bool: True if image is a base64 encoded string, False otherwise.
        """
        try:
            base64.b64decode(encoded_string)
            return True
        except Exception:
            return False

    def _url_to_base64(self) -> str:
        """
        Convert the image to a base64 string.

        Returns:
            str: The base64 encoded image.
        """
        encoded_image = base64.standard_b64encode(
            httpx.get(self.url).content).decode("utf-8")
        # return f"data:{self.image_mime_type};base64,{encoded_image}"
        return encoded_image

    def _file_to_base64(self) -> str:
        """
        Convert the image to a base64 string.

        Returns:
            str: The base64 encoded image.
        """
        with open(self.file_path, "rb") as image_file:
            binary_data = image_file.read()
            base64_encoded = base64.b64encode(binary_data).decode('utf-8')
            data_url = f"data:{self.image_mime_type};base64,{base64_encoded}"
            return data_url
    
    def _get_image_mime_type(base64_string):
        """
        Get the MIME type of the image from the base64 encoded string.

        Returns:
            str: The MIME type of the image.
        """
        # Decode the base64 string to bytes
        image_data = base64.b64decode(base64_string)

        # Dictionary of file signatures and their corresponding MIME types
        signatures = {
            b'\xFF\xD8\xFF': "image/jpeg",
            b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A': "image/png",
            b'GIF87a': "image/gif",
            b'GIF89a': "image/gif",
            b'RIFF': "image/webp"  # WEBP starts with "RIFF" header
        }

        # Check the first few bytes against known signatures
        for signature, mime_type in signatures.items():
            if image_data.startswith(signature):
                return mime_type

        return None  # Return None if no matching type is found


    def _is_image_type_supported(self, image_type: str) -> bool:
        """
        Check if the image type is supported.

        Returns:
            bool: True if the image type is supported, False otherwise.
        """
        extensions = self.IMAGE_TYPE_EXTENSION_MAP.values()
        mime_types = self.IMAGE_TYPE_EXTENSION_MAP.keys()

        return image_type in extensions or image_type in mime_types

    def _get_image_type_from_url(self, url: str) -> str:
        """
        Get the type of the image from the URL string.

        Returns:
            str: The type of the image.
        """
        # get the extension of the url
        extension = os.path.splitext(url)[1]
        # map the extension to the image type
        mime_type = self.IMAGE_TYPE_EXTENSION_MAP.get(extension)
        if not self._is_image_type_supported(mime_type):
            raise ValueError(f"Image type {mime_type} is not supported.")
        return extension, mime_type

    def _get_image_type_from_file(self, file_path: str) -> str:
        """
        Get the type of the image from the file path.

        Returns:
            str: The type of the image.
        """
        # get the file extension
        extension = os.path.splitext(file_path)[1]
        # map the extension to the image type
        image_type = self.IMAGE_TYPE_EXTENSION_MAP.get(extension)
        if not self._is_image_type_supported(image_type):
            raise ValueError(f"Image type {image_type} is not supported.")
        return image_type

    def _get_image_type_from_base64(self, base64_string: str) -> str:
        """
        Get the type of the image from the base64 encoded string.

        Returns:
            str: The type of the image.
        """
        # get the mime type of the base64 string
        mime_type, _ = guess_type(base64_string)
        if not self._is_image_type_supported(mime_type):
            raise ValueError(f"Image type {mime_type} is not supported.")
        return mime_type

    def _add_mime_type_to_base64(self, mime_type: str, base64_string: str) -> str:
        """
        Add the MIME type to the base64 encoded string.

        Returns:
            str: The base64 encoded string with the MIME type.
        """
        return f"data:{mime_type};base64,{base64_string}"
