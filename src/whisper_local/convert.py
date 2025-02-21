import os
from pydub import AudioSegment


def convert_to_supported_format(input_file: str, target_format: str = "mp3") -> str:
    """
    Convert the audio file to the specified target format using pydub.

    Args:
        input_file (str): Path to the source audio file.
        target_format (str): The desired format, e.g. 'mp3'. Default is 'mp3'.

    Returns:
        str: Path to the newly created file in the specified format.

    Raises:
        Exception: If the conversion fails.
    """
    base, ext = os.path.splitext(input_file)
    output_file = base + f".{target_format}"

    try:
        # Load audio file (pydub automatically detects format from extension)
        audio = AudioSegment.from_file(input_file)

        # Export to target format
        audio.export(
            output_file,
            format=target_format,
            parameters=["-ac", "2"]  # Force stereo output
        )

        return output_file

    except Exception as e:
        raise Exception(f"Error converting file: {str(e)}")
