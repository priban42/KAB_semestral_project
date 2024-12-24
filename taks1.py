import utils
from utils import text_to_numpy, extract_words_from_wikipedia_file, substitution_keys_iterator_wiki, key_table, numpy_to_text, text_is_english_fast, plot_frequency_analysis, frequency_analysis, index_of_coincidence
import numpy as np
import heapq

def main():
    text = "QSZBGDOZJTQSZQVJJTQSZGVDFEBHOSDHIBHSDHIADNEQJQSZKDMEBHOFJQNDGZQJSZMXJRHOFJUZMPRHIZMDPRGGZMPRHSZMVBQSSBPSDHIEZMNSBZTPQRTTZIBHSZMGJRQSPSZMDBPZPSZMSDHIQJQSZPGBFBHOAZHZUJFZHQDQQZHIDHQDHIPDXPSRHOSRSORQS"
    wikipedia_word_set = extract_words_from_wikipedia_file('Human rights in Denmark.htm')
    text_arr = text_to_numpy(text)
    ic = index_of_coincidence(text_arr)
    print(f"ic:{ic:04f}")
    # plot_frequency_analysis(frequency_analysis(text_arr), utils.english_freq)
    with open("10000-most-common-words.txt", "r") as file:
        top_1000_words = [line.strip() for line in file]
        top_1000_words = [word.upper() for word in top_1000_words if len(word) >= 2]
        top_1000_words = top_1000_words

    top_1000_words_set = set(top_1000_words)
    top_1000_words_set_extended = set(top_1000_words)
    for word in top_1000_words:
        for j in range(1, len(word)):
            top_1000_words_set_extended.add(word[:j])

    N = 100
    top_N = []
    for key in substitution_keys_iterator_wiki(wikipedia_word_set):
        table = key_table(key)
        result = np.argsort(table)[text_arr]
        dist = text_is_english_fast(numpy_to_text(result), top_1000_words_set, top_1000_words_set_extended)
        key_text = numpy_to_text(key)
        result_bundle = [float(dist), key_text, numpy_to_text(result)]
        if len(top_N) < N:
            heapq.heappush(top_N, result_bundle)
        else:
            if dist > top_N[0][0]:
                heapq.heapreplace(top_N, result_bundle)

    top_N_sorted = sorted(top_N, key=lambda x: -x[0])
    for a in top_N_sorted[:5]:
        print(f"{a}")

if __name__ == "__main__":
    main()