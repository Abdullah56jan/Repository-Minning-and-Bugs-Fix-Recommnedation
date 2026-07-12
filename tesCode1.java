public class StatusManager {
    public boolean isActive(String status) {
        // BUG: If 'status' is null, calling .equals() on it
        // will immediately throw a NullPointerException!
        if (status.equals("ACTIVE")) {
            return true;
        }
        return false;
    }
}