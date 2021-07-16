package xyz.hyhy.scai;

public class Main extends MainClass {
    public static int[] nums;
    public static double count = 0;


    public static void main(String[] args) {
//        nums = new int[15];
//        for (int i = 0; i < 15; i++) {
//            nums[i] = i;
//        }
//        pl(0);
//        for (int i = 2; i <= 15; i++) {
//            count /= i;
//        }
//        System.out.println(count);
        int c = 52;
        long count = 1;
        int m = 1;
        for (int i = 0; i < 13; i++) {
            count *= c--;
            count /= m++;
        }
        System.out.println(count /4);
    }

    public static void pl(int k) {
        if (k == nums.length) {
            for (int i = 1; i < 15; i++) {
                if (((nums[i] + nums[i - 1]) & 1) == 1)
                    count++;
            }
            return;
        }
        for (int i = k; i < nums.length; i++) {
            int temp = nums[k];
            nums[k] = nums[i];
            nums[i] = temp;
            pl(k + 1);
            temp = nums[k];
            nums[k] = nums[i];
            nums[i] = temp;
        }
    }
}
