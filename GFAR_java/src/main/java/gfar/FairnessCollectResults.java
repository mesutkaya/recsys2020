package gfar;

import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import es.uam.eps.ir.ranksys.metrics.RecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.SystemMetric;
import es.uam.eps.ir.ranksys.metrics.basic.AverageRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.basic.NDCG;
import es.uam.eps.ir.ranksys.metrics.rank.LogarithmicDiscountModel;
import es.uam.eps.ir.ranksys.metrics.rel.BinaryRelevanceModel;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;
import gfar.metrics.*;
import gfar.util.GFARPreferenceReader;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Paths;
import java.util.*;

import static org.ranksys.formats.parsing.Parsers.lp;


public class FairnessCollectResults {

    static final String[] algorithmNames = {"AVG", "FAI", "XPO", "GreedyLM", "SPGreedy", "GFAR"};
    static final String[] reRankStrings = {"", "", "", "_GreedyLM_1.0", "_SPGreedy_1.0", "_GFAR_1.0"};


    static final String[] groupTypes = {"random", "sim", "div"};
    static final String[] datasets = {"ml1m", "kgrec"};
    //static final String groupType = "div";

    public static void main(String[] args) throws Exception {
        String PROJECT_FOLDER = Paths.get(System.getProperty("user.dir")).getParent().toString();

        for (String dataset:datasets) {
            System.out.println("Results for the " + dataset + " dataset");
            System.out.println("=================================================");
            String DATA_PATH = PROJECT_FOLDER + "/data/" + dataset + "/";
            String[] folds = {"1", "2", "3", "4", "5"};
            //String[] folds = {""};
            //int[] groupSizes = {2, 3, 4, 5, 6, 7, 8};
            int[] groupSizes = {8};
            int cutoff = 20;
            double relevanceThreshold = dataset.equals("kgrec") ? 1.0 : 4.0;
            String baseFileName = dataset.equals("kgrec") ? "mf_230_1.0" : "mf_30_1.0";

            String fileNameBase;

            LogarithmicDiscountModel ldisc = new LogarithmicDiscountModel();

            for (String groupType : groupTypes) {
                System.out.println("Results for group Type: " + groupType);
                System.out.println("algorithm,groupSize,zRecall,recall,recallmin,recallminmax,ndcg,ndcgmin,ndcgminmax,dfh,dfhmin,dfhminmax");
                for (int i = 0; i < algorithmNames.length; i++) {
                    String algorithm = algorithmNames[i];
                    switch (algorithm) {
                        case "LM":
                            fileNameBase = baseFileName + "_lm_" + groupType + "_group_";
                            break;
                        case "FAI":
                            fileNameBase = baseFileName + "_fai_" + groupType + "_group_";
                            break;
                        case "XPO":
                            fileNameBase = baseFileName + "_xpo_" + groupType + "_group_";
                            break;
                        case "MAX":
                            fileNameBase = baseFileName + "_max_" + groupType + "_group_";
                            break;
                        default:
                            fileNameBase = baseFileName + "_avg_" + groupType + "_group_";
                            break;
                    }
                    String reRank = reRankStrings[i];
                    Map<String, Map<String, ArrayList<Double>>> results = new HashMap<>();
                    for (int groupSize : groupSizes) {
                        String fileName = fileNameBase + groupSize;
                        String groupsFilePath = DATA_PATH + groupType + "_group_" + groupSize;
                        Map<Long, List<Long>> groups = loadGroups(groupsFilePath);
                        int numGroups = groups.size();

                        for (String fold : folds) {
                            String testDataPath = DATA_PATH + fold + "/test.csv";
                            PreferenceData<Long, Long> testData = SimplePreferenceData.load(GFARPreferenceReader.get().read(testDataPath, lp, lp));
                            BinaryRelevanceModel<Long, Long> binRel = new BinaryRelevanceModel<>(false, testData, relevanceThreshold);
                            RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<Long, Long>(lp, lp);
                            String recIn = DATA_PATH + fold + "/" + fileName + reRank;
                            Map<String, SystemMetric<Long, Long>> sysMetrics = new HashMap<>();
                            Map<String, RecommendationMetric<Long, Long>> recMetrics = new HashMap<>();

                            // Zero Recall Metric
                            recMetrics.put("zerorecall", new ZeroRecallGroupFairness<Long, Long, Long>(cutoff, groups, binRel));

                            // Recall Metrics (mean, min, minmax)
                            recMetrics.put("recall", new RecallGroup<>(cutoff, groups, binRel));
                            recMetrics.put("recallmin", new RecallGroupFairness<Long, Long, Long>(cutoff, groups, binRel, "MIN"));
                            recMetrics.put("recallminmax", new RecallGroupFairness<Long, Long, Long>(cutoff, groups, binRel, "MIN-MAX"));

                            //NDCG Metrics (mean, min, minmax)
                            recMetrics.put("ndcg", new NDCGGroup<>(new NDCG.NDCGRelevanceModel<>(false, testData, relevanceThreshold), cutoff, ldisc, groups));
                            recMetrics.put("ndcgmin", new NDCGGroupFairness<>(new NDCG.NDCGRelevanceModel<>(false, testData, relevanceThreshold), cutoff, ldisc, groups, "MIN"));
                            recMetrics.put("ndcgminmax", new NDCGGroupFairness<>(new NDCG.NDCGRelevanceModel<>(false, testData, relevanceThreshold), cutoff, ldisc, groups, "MIN-MAX"));

                            // DFH Metrics (mean, min, minmax)
                            recMetrics.put("dfh", new DiscountedFirstHit<>(cutoff, ldisc, groups, binRel));
                            recMetrics.put("dfhmin", new DiscountedFirstHitFairness<>(cutoff, ldisc, groups, binRel, "MIN"));
                            recMetrics.put("dfhminmax", new DiscountedFirstHitFairness<>(cutoff, ldisc, groups, binRel, "MIN-MAX"));

                            recMetrics.forEach((name, metric) -> sysMetrics.put(name, new AverageRecommendationMetric<>(metric, numGroups)));
                            format.getReader(recIn).readAll().forEach(rec -> sysMetrics.values().forEach(metric -> metric.add(rec)));

                            if (!results.containsKey(Integer.toString(groupSize))) {
                                Map<String, ArrayList<Double>> temp_results = new HashMap<>();
                                sysMetrics.forEach((name, metric) -> {

                                    ArrayList<Double> temp = new ArrayList<>();
                                    temp.add(metric.evaluate());
                                    temp_results.put(name, temp);
                                });
                                results.put(Integer.toString(groupSize), temp_results);
                            } else {
                                sysMetrics.forEach((name, metric) -> {
                                    results.get(Integer.toString(groupSize)).get(name).add(metric.evaluate());
                                });
                            }
                        }
                    }
                    results.forEach((size, stringArrayListMap) -> {
                        double zerorecall = stringArrayListMap.get("zerorecall").stream().mapToDouble(a -> a).average().getAsDouble();

                        double recall = stringArrayListMap.get("recall").stream().mapToDouble(a -> a).average().getAsDouble();
                        double recallmin = stringArrayListMap.get("recallmin").stream().mapToDouble(a -> a).average().getAsDouble();
                        double recallminmax = stringArrayListMap.get("recallminmax").stream().mapToDouble(a -> a).average().getAsDouble();

                        double ndcg = stringArrayListMap.get("ndcg").stream().mapToDouble(a -> a).average().getAsDouble();
                        double ndcgmin = stringArrayListMap.get("ndcgmin").stream().mapToDouble(a -> a).average().getAsDouble();
                        double ndcgminmax = stringArrayListMap.get("ndcgminmax").stream().mapToDouble(a -> a).average().getAsDouble();

                        double dfh = stringArrayListMap.get("dfh").stream().mapToDouble(a -> a).average().getAsDouble();
                        double dfhmin = stringArrayListMap.get("dfhmin").stream().mapToDouble(a -> a).average().getAsDouble();
                        double dfhminmax = stringArrayListMap.get("dfhminmax").stream().mapToDouble(a -> a).average().getAsDouble();

                        String outStr = algorithm + "," + size + "," + zerorecall + "," + recall + "," + recallmin + "," + recallminmax + "," +
                                ndcg + "," + ndcgmin + "," + ndcgminmax
                                + "," + dfh + "," + dfhmin + "," + dfhminmax;
                        System.out.println(outStr);
                    });
                }
                System.out.println("=================================================");
            }
            System.out.println("=================================================");
        }
    }

    /**
     * Loads the ids of the users for each group from a file (for synthetic groups)!
     *
     * @param filePath
     * @return
     */
    public static Map<Long, List<Long>> loadGroups(String filePath) {
        Scanner s = null;
        try {
            s = new Scanner(new File(filePath));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        Map<Long, List<Long>> groups = new HashMap<>();

        if (s != null) {
            while (s.hasNext()) {
                List<Long> group_members = new ArrayList<>();
                String[] parsedLine = s.nextLine().split("\t");
                long id = Long.parseLong(parsedLine[0]);
                for (int i = 1; i < parsedLine.length; i++) {
                    group_members.add(Long.parseLong(parsedLine[i]));
                }
                groups.put(id, group_members);
            }
        }
        if (s != null) {
            s.close();
        }
        return groups;
    }
}
