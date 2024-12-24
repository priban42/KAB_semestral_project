import utils
from utils import text_to_numpy, numpy_to_text, plot_frequency_analysis, frequency_analysis, index_of_coincidence, get_all_factor_pairs, transpose, find_best_shift, roll_array, affine_find_key


def main():
    text = "CYVIRVULKBRTLGKLYVMVIVIVMZVGPXQYLRLVIUPKLPVGVTVBIPGKLYUZZYGFZCLLGZCPLVGMLGFLLIGKLRPQZZIBITXQIXGPVIUQBILVQQALPVXDLVIULYPVGEUVBYNHXLLIBDLDYLVR"
    text_arr = text_to_numpy(text)
    ic = index_of_coincidence(text_arr)
    print(f"ic:{ic:04f}")
    # plot_frequency_analysis(frequency_analysis(text_arr), utils.english_freq)
    shapes = get_all_factor_pairs(len(text_arr))
    result = affine_find_key(text_arr)
    print(result)
    pass


if __name__ == "__main__":
    main()