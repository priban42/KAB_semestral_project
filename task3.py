import utils
from utils import text_to_numpy, numpy_to_text, plot_frequency_analysis, frequency_analysis, index_of_coincidence, get_all_factor_pairs, transpose, find_best_shift, roll_array


def main():
    text = "PPVIFRBVZAPWAYHDPUYUOULXFXAXPXPXWXZXZXUXLXLXLXDXRZNYJVSBNILPLYVUHHPAHYXUXYXYXZXHXYXUXVXBXOXDXLXSTUYHFHQUAOAPMUUOLKLOLKHXHXLXLXUXLXVXAXNXAXFXYXHX"
    text_arr = text_to_numpy(text)
    ic = index_of_coincidence(text_arr)
    print(f"ic:{ic:04f}")
    # plot_frequency_analysis(frequency_analysis(text_arr), utils.english_freq)
    shapes = get_all_factor_pairs(len(text_arr))
    english_freq_roll = roll_array(utils.english_freq)

    for shape1 in shapes[1:-1]:
        for shape2 in shapes[1:-1]:
            result = transpose(transpose(text_arr, shape1), shape2)
            text = numpy_to_text(result)
            if text[-25:] == "X"*25:
                freq = frequency_analysis(text_to_numpy(text))
                key = find_best_shift(freq, english_freq_roll)
                print(f"transpose 1:{shape1}, transpose 2:{shape2}, shift:{key}, {numpy_to_text((result - key)%26)}")
    pass


if __name__ == "__main__":
    main()