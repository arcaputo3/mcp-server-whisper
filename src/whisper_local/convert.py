import os
from pydub import AudioSegment

def convert_to_supported_format(input_file: str, target_format: str = "mp3") -> str:
    """
    Convert the audio file to the specified target format using pydub.
    By default, converts to 'mp3'.
    
    Args:
        input_file (str): Path to the source audio file.
        target_format (str): The desired format, e.g. 'mp3'. Default is 'mp3'.

    Returns:
        str: Path to the newly created file in the specified format.

    Raises:
        Exception: If the conversion fails.
    """
    base, _ = os.path.splitext(input_file)
    output_file = base + f".{target_format}"

    try:
        # Load audio file (pydub automatically detects format from extension)
        audio = AudioSegment.from_file(input_file)
        # Export to the target format
        audio.export(
            output_file,
            format=target_format,
            parameters=["-ac", "2"]  # Force stereo output
        )
        return output_file
    except Exception as e:
        raise Exception(f"Error converting file: {str(e)}")


def compress_mp3_file(mp3_file_path: str, out_sample_rate: int = 11025) -> str:
    """
    Compress the mp3 file by downsampling to out_sample_rate (default 11025).
    Creates a new output file prefixed with 'compressed_'.
    
    Args:
        mp3_file_path (str): Path to the mp3 file.
        out_sample_rate (int): Desired sample rate, default 11025.

    Returns:
        str: The path to the newly compressed MP3 file (e.g., 'compressed_<basename>.mp3').
    """
    if not mp3_file_path.lower().endswith(".mp3"):
        raise Exception("compress_mp3_file() called on a file that is not .mp3")

    basename = os.path.basename(mp3_file_path)
    name_no_ext, _ = os.path.splitext(basename)
    output_file_path = f"compressed_{name_no_ext}.mp3"

    print(f"\n[Compression] Original file: {mp3_file_path}")
    print(f"[Compression] Output file:   {output_file_path}")

    try:
        audio_file = AudioSegment.from_file(mp3_file_path, "mp3")
        original_frame_rate = audio_file.frame_rate
        print(f"[Compression] Original frame rate: {original_frame_rate}, converting to {out_sample_rate}.")

        # Export the compressed file
        audio_file.export(
            output_file_path,
            format="mp3",
            parameters=["-ar", str(out_sample_rate)]
        )
        return output_file_path
    except Exception as e:
        raise Exception(f"Error compressing mp3 file: {str(e)}")


def maybe_compress_file(input_file: str, max_mb: int = 25) -> str:
    """
    Check if 'input_file' is above 'max_mb' megabytes. If so, compress it by:
      1) Converting to MP3 (if not already .mp3).
      2) Downsampling to 11025 Hz (via compress_mp3_file).
    Returns the path to the (possibly compressed) file.
    
    If the file is still over max_mb after compression, we do not attempt further passes.
    It is the caller's responsibility to handle the size if it remains above the threshold.
    
    Args:
        input_file (str): Path to the audio file.
        max_mb (int): Threshold in MB; default 25MB.

    Returns:
        str: Path to the file (compressed if needed).
    """
    file_size = os.path.getsize(input_file)
    threshold_bytes = max_mb * 1024 * 1024

    if file_size <= threshold_bytes:
        return input_file  # No compression needed

    print(f"\n[maybe_compress_file] File '{input_file}' size > {max_mb}MB. Attempting compression...")

    # 1) If not mp3, convert to mp3
    if not input_file.lower().endswith(".mp3"):
        try:
            input_file = convert_to_supported_format(input_file, target_format="mp3")
        except Exception as e:
            raise Exception(f"[maybe_compress_file] Error converting to MP3: {str(e)}")

    # 2) Compress by downsampling
    try:
        compressed_path = compress_mp3_file(input_file, 11025)
    except Exception as e:
        raise Exception(f"[maybe_compress_file] Error compressing MP3 file: {str(e)}")

    # Overwrite input_file to refer to compressed file
    input_file = compressed_path
    new_size = os.path.getsize(input_file)
    print(f"[maybe_compress_file] Compressed file size: {new_size} bytes")

    return input_file