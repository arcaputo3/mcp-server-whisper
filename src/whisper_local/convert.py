import os
import asyncio
from pydub import AudioSegment
from typing import Optional


async def convert_to_supported_format(
    input_file: str, target_format: str = "mp3"
) -> str:
    """
    Async version of audio file conversion using pydub.
    Ensures the output filename is base + .{target_format}.
    """
    base, ext = os.path.splitext(input_file)
    output_file = base + f".{target_format}"

    try:
        audio = await asyncio.to_thread(AudioSegment.from_file, input_file)
        await asyncio.to_thread(
            audio.export, output_file, format=target_format, parameters=["-ac", "2"]
        )
        return output_file
    except Exception as e:
        raise Exception(f"Error converting file: {str(e)}")


async def compress_mp3_file(mp3_file_path: str, out_sample_rate: int = 11025) -> str:
    """
    Downsample an existing mp3. Return a file named 'compressed_{original_stem}.mp3'.
    """
    if not mp3_file_path.lower().endswith(".mp3"):
        raise Exception("compress_mp3_file() called on a file that is not .mp3")

    basename = os.path.basename(mp3_file_path)
    name_no_ext, _ = os.path.splitext(basename)
    output_file_path = f"compressed_{name_no_ext}.mp3"

    print(f"\n[Compression] Original file: {mp3_file_path}")
    print(f"[Compression] Output file:   {output_file_path}")

    try:
        audio_file = await asyncio.to_thread(
            AudioSegment.from_file, mp3_file_path, "mp3"
        )
        original_frame_rate = audio_file.frame_rate
        print(
            f"[Compression] Original frame rate: {original_frame_rate}, converting to {out_sample_rate}."
        )
        await asyncio.to_thread(
            audio_file.export,
            output_file_path,
            format="mp3",
            parameters=["-ar", str(out_sample_rate)],
        )
        return output_file_path
    except Exception as e:
        raise Exception(f"Error compressing mp3 file: {str(e)}")


async def maybe_compress_file(input_file: str, max_mb: int = 25) -> str:
    """
    If file is above {max_mb}, convert to mp3 if needed, then compress.
    Return the final compressed_{stem}.mp3 path if compression happens,
    otherwise return the original path.
    """
    file_size = os.path.getsize(input_file)
    threshold_bytes = max_mb * 1024 * 1024

    if file_size <= threshold_bytes:
        return input_file  # No compression needed

    print(
        f"\n[maybe_compress_file] File '{input_file}' size > {max_mb}MB. Attempting compression..."
    )

    # If not mp3, convert
    if not input_file.lower().endswith(".mp3"):
        try:
            input_file = await convert_to_supported_format(input_file, "mp3")
        except Exception as e:
            raise Exception(f"[maybe_compress_file] Error converting to MP3: {str(e)}")

    # now downsample
    try:
        compressed_path = await compress_mp3_file(input_file, 11025)
    except Exception as e:
        raise Exception(f"[maybe_compress_file] Error compressing MP3 file: {str(e)}")

    new_size = os.path.getsize(compressed_path)
    print(f"[maybe_compress_file] Compressed file size: {new_size} bytes")
    return compressed_path
