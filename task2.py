import utils
from utils import text_to_numpy, numpy_to_text, plot_frequency_analysis, frequency_analysis, index_of_coincidence, vigenere_find_key

def main():
    text = "QZXSIKGDLMFFMUBUDYKUFRRLASSTRSSAXADIANNHFUDTSCFQCDXIYMVTDIJVWFPLJJKFZMEMKRHVVTUDLMNXOVWGZYLENZVBTWFCLYLRMALHMNQLVZWEXWUNWWXMEEVQMICIMOLMEZTJAMDTIGTZVKBSIAVIIRZITRJENKYMZPHICUVVAMYWQIJPPYZLRIFPIUAQXMLVIRKUQEIIADSWBBZJNQNHZ"
    text_arr = text_to_numpy(text)
    ic = index_of_coincidence(text_arr)
    print(f"ic:{ic:04f}")
    for key_length in [4, 5, 6, 7, 8, 9, 10]:
        key, result = vigenere_find_key(text_arr, key_length)
        print(f"{key_length}: {numpy_to_text((26-key)%26)}, {result}")

if __name__ == "__main__":
    main()