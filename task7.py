import utils
from utils import text_to_numpy, numpy_to_text, index_of_coincidence, get_all_factor_pairs, transpose, roll_array, affine_find_key, plot_frequency_analysis


def main():
    text = "MPQPTDDHLJLPUWRTUQYRWVOQDNJJJLLPNDJJMBATDEHQMQEXRHAWGEMJUNHXUJMJXGPRJXWXLJUDJWMRWDDXXQJVUPPMJNEXOTPRDQQUDRJXDDWLOMYJWMAX"
    text_arr = text_to_numpy(text)
    ic = index_of_coincidence(text_arr)
    print(f"ic:{ic:04f}")
    shapes = get_all_factor_pairs(len(text_arr))
    english_freq_roll = roll_array(utils.english_freq)
    for shape1 in shapes[1:-1]:
        result = transpose(text_arr, shape1)
        text = numpy_to_text(result)
        if text[-6:] == "X"*6:
            # print(f"transpose 1:{shape1}, transpose 2:{shape2}, shift:{key}, {numpy_to_text((result - key)%26)}")
            result_affine = affine_find_key(result)
            print(f"transpose 1:{shape1}, {result_affine}")
            # plot_frequency_analysis([utils.english_freq, frequency_analysis(text_arr), frequency_analysis(text_to_numpy(result_affine[2][0][1]))],
            #                         legend=["anglicky text", "ŠT", "OT"], styles=["b--", "g-", "r-"])
    pass


if __name__ == "__main__":
    main()