import utils
from utils import text_to_numpy, numpy_to_text, plot_frequency_analysis, frequency_analysis, index_of_coincidence, get_all_factor_pairs, transpose, find_best_shift, roll_array, text_is_english_fast
import numpy as np
from itertools import permutations
import heapq

def main():
    text = "BRIGHTREDBIGJOKEFRANNIEHESAIDUNCERTAINLYNOJOKEHEKEPTLOOKINGATHERAFTERAWHILETHEYSTARTEDWALKINGAGAINASTHEYCROSSEDTHEPARKINGLOTGUSCAMEOUTANDWAVEDTOTHEMFRANNIEWAVEDBACKXXXXX"
    text_arr = text_to_numpy(text)
    ic = index_of_coincidence(text_arr)
    print(f"ic:{ic:04f}")
    print(len(text))
    # plot_frequency_analysis([utils.english_freq, frequency_analysis(text_arr)], legend=["anglicky text", "ŠT"], styles=["b--", "g-"])
    shapes = get_all_factor_pairs(len(text_arr))

    for shape1 in shapes[1:-1]:
        for shape2 in shapes[1:-1]:
            for perm1 in permutations(range(shape1[0])):
                for perm2 in permutations(range(shape1[0])):
                    result = transpose(transpose(text_arr, shape1, perm=np.array(perm1)), shape2, perm=np.array(perm2))
                    if np.all(result == text_arr):
                        print(f"{shape1}, {shape2}, {perm1}")


if __name__ == "__main__":
    main()