package gfar;

import java.nio.file.Paths;

public class testpath {
    public static void main(String[] args) {
        System.out.println(Paths.get(System.getProperty("user.dir")).getParent().toString());
    }
}
