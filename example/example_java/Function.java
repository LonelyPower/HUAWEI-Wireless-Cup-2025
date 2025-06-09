import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Function {
    public static void func() {
        Random random = new Random();
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            int operatorId = random.nextInt(11);
            int tilingId = random.nextInt(11);
            int nodeIdInOperator = random.nextInt(11);
            int nodeStartTime = random.nextInt(11);
            int processorIdForNode = random.nextInt(11);
            List<Integer> scheduleNode = new ArrayList<>();
            scheduleNode.add(operatorId);
            scheduleNode.add(tilingId);
            scheduleNode.add(nodeIdInOperator);
            scheduleNode.add(nodeStartTime);
            scheduleNode.add(processorIdForNode);
            ans.add(scheduleNode);
        }
        System.out.println(ans);
    }
}