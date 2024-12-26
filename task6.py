import utils
from utils import text_to_numpy, numpy_to_text, plot_frequency_analysis, frequency_analysis, index_of_coincidence, get_all_factor_pairs, transpose, find_best_shift, roll_array, affine_find_key


def main():
    text = "CYVIRVULKBRTLGKLYVMVIVIVMZVGPXQYLRLVIUPKLPVGVTVBIPGKLYUZZYGFZCLLGZCPLVGMLGFLLIGKLRPQZZIBITXQIXGPVIUQBILVQQALPVXDLVIULYPVGEUVBYNHXLLIBDLDYLVR"
    text_arr = text_to_numpy(text)
    ic = index_of_coincidence(text_arr)
    print(f"ic:{ic:04f}")
    shapes = get_all_factor_pairs(len(text_arr))
    result = affine_find_key(text_arr)
    # plot_frequency_analysis(
    #     [utils.english_freq, frequency_analysis(text_arr), frequency_analysis(text_to_numpy(result[2][0][1]))],
    #     legend=["anglicky text", "ŠT", "OT"], styles=["b--", "g-", "r-"])
    print(result)
    pass


if __name__ == "__main__":
    main()