import os
import unicodedata
import pandas

wav_dir = r'D:\DAYDAYUP\ASR\data\corpus\syn_continuous_digit_soft_noise_3_20\test\syn_wav'
labels_file_path = r'D:\DAYDAYUP\ASR\data\corpus\syn_continuous_digit_soft_noise_3_20\test/syn_labels_unit_test.txt'
output_csv_file_path = r'D:\DAYDAYUP\ASR\data\corpus\syn_continuous_digit_soft_noise_3_20\test/syn_csv_unit_test.csv'


def gen_csv(wav_dir, labels_file_path, output_csv_file_path):
    files = []
    with open(labels_file_path, 'r') as labels_file:
        for line in labels_file:
            seqid, transcript = line.split(" ", 1)
            transcript = unicodedata.normalize("NFKD", transcript).encode(
                "ascii", "ignore").decode("ascii", "ignore").strip().lower()
            wav_file_path = os.path.join(wav_dir, seqid + ".wav")
            wav_file_size = os.path.getsize(wav_file_path)
            files.append((os.path.abspath(wav_file_path), wav_file_size, transcript))

    df = pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])
    df.to_csv(output_csv_file_path, index=False, sep="\t")


gen_csv(wav_dir, labels_file_path, output_csv_file_path)
