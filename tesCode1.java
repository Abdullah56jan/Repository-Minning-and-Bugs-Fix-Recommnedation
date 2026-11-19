public int findMax(int[] numbers) {
    int max = numbers[0];
    // BUG: Using <= causes the loop to check an index that doesn't exist
    for (int i = 1; i <= numbers.length; i++) {
        if (numbers[i] > max) {
            max = numbers[i];
        }
    }
    return max;
}