import java.util.*;

public class DasherPaySolution {

    // --- 1. 基础模型 ---
    enum Action { ACCEPT, ARRIVE, PICKUP, FULFILL }

    static class Event {
        int time;
        String orderId;
        Action action;
        Event(int time, String orderId, Action action) {
            this.time = time;
            this.orderId = orderId;
            this.action = action;
        }
    }

    // --- 2. 定义 Mock API 接口 (The Contract) ---
    // 为了方便放在一个文件里运行，我把它定义在类内部
    interface DasherEventClient {
        List<Event> fetchEvents(String dasherId);
    }

    // --- 3. 核心业务逻辑类 (The Service) ---
    static class PayService {
        private static final double BASE_RATE_PER_MIN = 0.3;
        private static final Map<Action, Integer> ACTION_PRIORITY;
        static {
            Map<Action, Integer> map = new HashMap<>();
            map.put(Action.ACCEPT, 0);
            map.put(Action.ARRIVE, 1);
            map.put(Action.PICKUP, 2);
            map.put(Action.FULFILL, 3);
            ACTION_PRIORITY = Collections.unmodifiableMap(map);
        }

        private final DasherEventClient eventClient;

        // Dependency Injection
        public PayService(DasherEventClient client) {
            this.eventClient = client;
        }

        public double calculatePayForDasher(String dasherId) {
            // Step 1: 从 API 拉取数据
            List<Event> events = eventClient.fetchEvents(dasherId);

            if (events == null || events.size() < 2) return 0.0;

            // Step 2: 排序
            sortEventsStable(events);

            // Step 3: 计算
            Set<String> activeOrders = new HashSet<>();
            double pay = 0.0;

            for (int i = 0; i < events.size() - 1; i++) {
                Event cur = events.get(i);

                if (cur.action == Action.ACCEPT) {
                    activeOrders.add(cur.orderId);
                } else if (cur.action == Action.FULFILL) {
                    activeOrders.remove(cur.orderId);
                }

                int a = cur.time;
                int b = events.get(i + 1).time;
                double minutes = Math.max(0, b - a);
                
                if (!activeOrders.isEmpty()) {
                    pay += minutes * BASE_RATE_PER_MIN * activeOrders.size();
                }
            }
            // 保留两位小数
            return Math.round(pay * 100.0) / 100.0;
        }

        private void sortEventsStable(List<Event> events) {
            Collections.sort(events, new Comparator<Event>() {
                @Override
                public int compare(Event a, Event b) {
                    if (a.time != b.time) return a.time - b.time;
                    return ACTION_PRIORITY.get(a.action) - ACTION_PRIORITY.get(b.action);
                }
            });
        }
    }

    // --- 4. Mock 实现 (Test Data) ---
    static class MockEventClient implements DasherEventClient {
        @Override
        public List<Event> fetchEvents(String dasherId) {
            List<Event> events = new ArrayList<>();
            // 测试用例：预期结果应该是 $14.4
            events.add(new Event(toMinutes("06:15"), "A", Action.ACCEPT));
            events.add(new Event(toMinutes("06:18"), "B", Action.ACCEPT));
            events.add(new Event(toMinutes("06:36"), "A", Action.FULFILL));
            events.add(new Event(toMinutes("06:45"), "B", Action.FULFILL));
            return events;
        }

        private int toMinutes(String time) {
            String[] parts = time.split(":");
            int h = Integer.parseInt(parts[0]);
            int m = Integer.parseInt(parts[1]);
            return h * 60 + m;
        }
    }

    // --- 5. Main 入口 ---
    public static void main(String[] args) {
        // 实例化 Mock 组件
        DasherEventClient mockClient = new MockEventClient();

        // 注入依赖
        PayService payService = new PayService(mockClient);

        // 运行
        System.out.println("Running Dasher Pay Calculation...");
        double actualPay = payService.calculatePayForDasher("dasher_123");

        System.out.println("------------------------------------------------");
        System.out.println("Expected Pay: $14.4");
        System.out.println("Actual Pay:   $" + actualPay);
        System.out.println("------------------------------------------------");
        
        if (Math.abs(actualPay - 14.4) < 0.001) {
            System.out.println("Test PASSED ✅");
        } else {
            System.out.println("Test FAILED ❌");
        }
    }
}