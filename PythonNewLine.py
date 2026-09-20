from pathlib import Path

def split_sentences_on_period(input_file: str, output_file: str):
    # Create paths for the input and output files.
    input_path = Path(input_file)
    output_path = Path(output_file)

    # Read the complete input file as UTF-8 text.
    text = input_path.read_text(encoding="utf-8")

    # Insert a newline after each period.
    processed_text = text.replace(".", ".\n")

    # Remove unnecessary whitespace from each line.
    processed_text = "\n".join(
        line.strip() for line in processed_text.splitlines()
    )

    # Write the processed text to the output file.
    output_path.write_text(processed_text, encoding="utf-8")


if __name__ == "__main__":
    # Input and output files are expected to be in the current working directory.
    split_sentences_on_period("Example.txt", "Example_newlines.txt")