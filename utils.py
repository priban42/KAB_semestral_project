from bs4 import BeautifulSoup
import numpy as np
from numpy.typing import NDArray, ArrayLike
import matplotlib.pyplot as plt
import re
from typing import Tuple, List
import math

english_freq = np.array([8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.772, 4.025, 2.406, 6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 0.978, 2.360, 0.150, 1.974, 0.074]) / 26


def text_to_numpy(text: str) -> NDArray[np.uint16]:
    upper_text = text.upper()
    arr = np.empty((len(upper_text)), dtype=np.uint16)
    for i in range(len(upper_text)):
        arr[i] = ord(upper_text[i]) - ord('A')
    return arr

def numpy_to_text(arr: NDArray) -> str:
    return (arr + ord('A')).astype(np.uint8).tobytes().decode("ascii")

def frequency_analysis(arr_text: NDArray, normalize: bool = True) -> NDArray:
    if arr_text.ndim == 1:
        extended_arr = np.hstack([np.arange(26), arr_text])
        letters, counts = np.unique(extended_arr, return_counts=True)
        frequency = counts - 1
        if normalize:
            return frequency/np.linalg.norm(frequency)
        else:
            return frequency
    else:
        extended_arr = np.hstack([np.tile(np.arange(26), (arr_text.shape[0], 1)), arr_text])
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=26), axis=1, arr=extended_arr)
        frequency = counts - 1
        if normalize:
            return frequency/np.linalg.norm(frequency, axis=1)[np.newaxis].T
        else:
            return frequency

def index_of_coincidence(arr_text: NDArray) -> float:
    text_freq = frequency_analysis(arr_text, normalize=False)
    N = arr_text.shape[0]
    return np.sum(text_freq * (text_freq - 1))/(N*(N-1))

def key_table(key:NDArray):
    """
    takes a word and generates a substitution table
    """
    table = np.zeros(26, dtype=np.uint8)
    table[:len(key)] = key
    used_set = set(key)
    a = 0
    for i in range(len(key), 26):
        while a in used_set:
            a += 1
        table[i] = a
        a += 1
    return table

def extract_words_from_wikipedia_file(file_path):
    """
    Reads a Wikipedia page file and creates a set of unique words from the text content.

    Args:
        file_path (str): The path to the Wikipedia file.

    Returns:
        set: A set of unique words from the file.
    """
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Parse the file with BeautifulSoup (for HTML files)
    soup = BeautifulSoup(content, 'html.parser')

    # Extract text content from the page
    text = soup.get_text()

    # Use regex to split text into words and normalize them to lowercase
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    # Return a set of unique words
    return set(words)

def remove_duplicate_characters(input_string):
    """
    Removes duplicate characters from a string while preserving the order of first occurrence.

    Args:
        input_string (str): The string to process.

    Returns:
        str: A new string with duplicates removed.
    """
    seen = set()
    result = []

    for char in input_string:
        if char not in seen:
            seen.add(char)
            result.append(char)

    return ''.join(result)

def text_is_english_fast(text: str, words_set:set, words_set_extended:set):
    count = 0
    word = ""
    best = 0
    for c in text + " ":
        word += c
        if word not in words_set_extended:
            count += best**2
            best = 0
            word = ""
        if word in words_set:
            best = len(word)
    return count

def substitution_keys_iterator_wiki(wiki_words):
    for word in wiki_words:
        for c in "abcdefghijklmnopqrstuvwxyz":
            yield text_to_numpy(remove_duplicate_characters(word+c))

def plot_frequency_analysis(frequencies: np.ndarray, english_freq, title: str = "Frequency Analysis"):
    """
    Plots a normalized frequency analysis using Matplotlib and saves it as a PDF.
    Also includes a baseline English frequency analysis as a dotted line.

    Parameters:
    frequencies (np.ndarray): A 1D NumPy array containing normalized frequency values.
    title (str): Title of the plot (default is 'Frequency Analysis').
    """
    # Ensure the input is a NumPy array
    frequencies = np.asarray(frequencies)

    # Generate indices for the x-axis (assuming bins or discrete frequency points)
    indices = np.arange(len(frequencies))

    # English letter frequencies (approximate for reference)

    # Extend English frequencies if necessary
    if len(english_freq) < len(frequencies):
        english_freq = np.pad(english_freq, (0, len(frequencies) - len(english_freq)), 'constant')

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(indices, english_freq[:len(frequencies)], 'r--', label='Average english text')
    plt.plot(indices, frequencies, color='blue', alpha=0.7, label='task')

    # Add baseline English frequencies as a dotted line


    # Add labels, title, and legend
    plt.xlabel('Letters')
    plt.ylabel('Normalized Frequency')
    plt.title(title)
    plt.legend()

    # Display grid for readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot as a PDF
    plt.savefig("frequency_analysis.pdf")

    # Show the plot
    plt.show()

def split_alternating(text_arr: NDArray, n: int) -> NDArray:
    """
    :param text_arr: [a, b, c, d, e, f, g]
    :param n: 2
    :return: [[a, c, e],
              [b, d, f]]
     (g does not fit so it is removed (len(text_arr)%2 = 1))
    """
    return np.reshape(text_arr[:text_arr.shape[0]-text_arr.shape[0]%n], (text_arr.shape[0]//n, n)).T

def find_best_shift(cipher_freq: NDArray, english_freq_roll:NDArray) -> int:
    """
    :param cipher_freq: frequency analysis of given cyphered text
    :param english_freq_roll: frequency analysis of the english language
    :return: alphabet shift that best aligns the two frequency analysis.
    """
    suma = np.sum((english_freq_roll-cipher_freq)**2, axis=1)
    best_shift = int(np.argmin(suma))
    # value = float(np.min(suma))
    return best_shift

def roll_array(arr: np.ndarray) -> np.ndarray:
    """
    :param arr: [a, b, c]
    :return: [[a, b, c],
              [b, c, a],
              [c, a, b]]
    """
    n = arr.shape[0]
    m = n  # number of rows in the new array
    result = np.zeros((m, n), dtype=arr.dtype)
    for i in range(m):
        result[i] = np.roll(arr, -i)
    return result

def decypher_split_arrays(split_arrays: NDArray, key: NDArray) -> NDArray:
    """
    :param split_arrays: [[a, b, c],
                          [e, f, g]]
    :param key: [1, 2]
    :return: [a+1, e+2, b+1, f+2, c+1, g+2] % 26
    """
    return np.reshape(((split_arrays + key[np.newaxis].T)%26).T, (split_arrays.shape[0]*split_arrays.shape[1]))

def vigenere_find_key(text_arr: NDArray, key_length) -> Tuple[NDArray, str]:
    """
    attempts to find the key for vigenere cipher
    """
    split_arrays = split_alternating(text_arr, key_length)
    key = np.zeros((key_length), dtype=np.uint16)
    english_freq_roll = roll_array(english_freq)
    for a in range(key_length):
        freq = frequency_analysis(split_arrays[a, :])
        key[a] = find_best_shift(freq, english_freq_roll)
    return key, numpy_to_text(decypher_split_arrays(split_arrays, key))

def transpose(text_arr:NDArray, shape:ArrayLike, rot=0, perm = None):
    table = text_arr.reshape(shape)
    if perm is not None:
        table = table[perm]
    return np.rot90(table, k=rot).T.reshape(len(text_arr))

def get_all_factor_pairs(n: int) -> NDArray:
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append([i, n//i])
    return np.array(factors)

def affine_encrypt(arr_text: NDArray, a: int, b: int) -> NDArray:
    return (arr_text*a + b)%26

def mod_inv(a: int, mod: int) -> int:
    """
    computes the multiplicative inversion a_inv of a so that (a * a_inv) % mod = 1
    """
    for i in range(1, 26):
        if a*i%mod == 1:
            return i

def affine_decrypt(arr_text: NDArray, a: int, b: int) -> NDArray:
    a_inv = mod_inv(a, 26)
    return a_inv*(arr_text.astype(np.int16) - b)%26

def affine_keys_iterator():
    """
    iterator function for affine cyphers. used to iterate over all possible a and b combinations.
    """
    for a in range(1, 26):
        if math.gcd(a, 26) == 1:
            for b in range(26):
                yield a, b

def affine_find_key(text_arr: NDArray):
    """
    attempts to find the key for the affine cipher
    """
    min_dist = 1000
    best_match = np.array([0])
    best_a = None
    best_b = None
    all_results = []
    for a, b in affine_keys_iterator():
        result = affine_decrypt(text_arr, a, b)
        result_text = numpy_to_text(result)
        # print(result_text)
        # freq = frequency_analysis(result_affine)
        # best_shift = find_best_shift(freq, english_freq_roll)
        # result = (result_affine + best_shift)%26
        # if numpy_to_text(result[:3]) == "THE":
        #     dist = 0
        dist = np.linalg.norm((frequency_analysis(result) - english_freq)) ** 2
        all_results.append([dist, result_text, a, b])
        # dist = -text_is_english2(numpy_to_text(result))

        # if text_is_english(numpy_to_text(result)) > 10:
        #     print(numpy_to_text(result))
        if dist < min_dist:
            min_dist = dist
            best_match = result
            best_a = a
            best_b = b

    return (best_a, best_b, sorted(all_results, key=lambda x: x[0]))
    # return {"a":best_a, "b":best_b, "text":numpy_to_text(best_match)}