import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();

        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            list.add(scanner.nextInt());
        }

        // 🔹 Step 1: Sort before binary search
        Collections.sort(list);

        int searchNumber = scanner.nextInt();

        int index = Collections.binarySearch(list, searchNumber);

        if (index >= 0) {
            System.out.println("Found at position: " + (index + 1));
        } else {
            int insertPos = -(index + 1);
            list.add(insertPos, searchNumber);
            System.out.println("Not found, added at position: " + (insertPos + 1));
        }

        System.out.println(list);
    }
}