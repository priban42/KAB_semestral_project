import utils
from utils import text_to_numpy, numpy_to_text, plot_frequency_analysis, frequency_analysis, index_of_coincidence, get_all_factor_pairs, transpose, find_best_shift, roll_array, text_is_english_fast
import numpy as np
from itertools import permutations
import heapq

def main():
    text = "DZDDSITAYOREESEXERXEQSMMNGLOOITLESNARXOEALELATDWPPOHDLDCXOEHDHRRAGKAENTCYLVUNHDNTUPESUSHAAALBYFPSISIIELONWSDHEIEELSEAASMRTOITDDFNIREXGUHIRGNYDWNOORUFNYXOENEAYNHOHOPKIDEVAX"
    text_arr = text_to_numpy(text)
    ic = index_of_coincidence(text_arr)
    print(f"ic:{ic:04f}")
    # plot_frequency_analysis([utils.english_freq, frequency_analysis(text_arr)], legend=["anglicky text", "ŠT"],
    #                         styles=["b--", "g-"])
    shapes = get_all_factor_pairs(len(text_arr))

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
    counter = 0
    for shape1 in shapes[1:-1]:
        for perm in permutations(range(shape1[0])):
            result = transpose(text_arr, shape1, perm= np.array(perm))
            # text = numpy_to_text(result)
            # if text[-6:] == "X"*6:
            if np.all(result[-6:] == np.array([23] * 6)):
                counter += 1
                dist = text_is_english_fast(numpy_to_text(result), top_1000_words_set, top_1000_words_set_extended)
                key_text = str(shape1) + str(perm)
                result_bundle = [float(dist), key_text, numpy_to_text(result)]
                if len(top_N) < N:
                    heapq.heappush(top_N, result_bundle)
                else:
                    if dist > top_N[0][0]:
                        heapq.heapreplace(top_N, result_bundle)
            if counter%1000 == 0:
                top_N_sorted = sorted(top_N, key=lambda x: -x[0])
                for a in top_N_sorted[:5]:
                    print(f"{a}")
    top_N_sorted = sorted(top_N, key=lambda x: -x[0])
    for a in top_N_sorted:
        print(f"{a}")


if __name__ == "__main__":
    main()