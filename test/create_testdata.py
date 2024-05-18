import os
from pydub import AudioSegment

def generate_test_data(input_dir, test_dir, duration_ms=765000):  # 12.75 minutes
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".m4a"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(test_dir, filename.replace(".m4a", "_test_short.m4a"))

            audio = AudioSegment.from_file(input_path, format="m4a")
            test_audio = audio[:duration_ms]
            test_audio.export(output_path, format="adts", codec="aac")

            print(f"Generated test data: {output_path}")

if __name__ == "__main__":
    input_directory = "./input"
    test_directory = "./test/data/"
    generate_test_data(input_directory, test_directory)