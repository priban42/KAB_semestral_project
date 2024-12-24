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
    # plot_frequency_analysis(frequency_analysis(text_arr), utils.english_freq)
    shapes = get_all_factor_pairs(len(text_arr))
    english_freq_roll = roll_array(utils.english_freq)

    for shape1 in shapes:
        for shape2 in shapes:
            result = transpose(transpose(text_arr, shape1), shape2)
            if np.all(result == text_arr):
                print(f"{shape1}, {shape2}")


if __name__ == "__main__":
    main()