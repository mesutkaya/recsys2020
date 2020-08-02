package gfar;

import es.uam.eps.ir.ranksys.core.Recommendation;
import gfar.aggregation.AverageScoreAggregationStrategy;
import gfar.aggregation.FairnessAggregationStrategy;
import gfar.aggregation.MaxSatisfactionAggStrategy;
import gfar.aggregation.XPO;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

import static org.ranksys.formats.parsing.Parsers.lp;
import java.nio.file.Paths;

/**
 * After generating the individual recommendations (ordered sets) for each individual, compute the group
 * recommendations using AVG, FAI and XPO by using this file.
 */

public class AggregateRecommedations {
    public static void main(String[] args) throws Exception {
        String PROJECT_FOLDER = Paths.get(System.getProperty("user.dir")).getParent().toString();
        String[] DataFolders = {PROJECT_FOLDER + "/data/ml1m/",
                PROJECT_FOLDER + "/data/kgrec/"};

        String[] strategies = {"XPO", "FAI", "AVG"};
        //String[] strategies = {"MAX"};

        String[] individualRecFileName = {"mf_30_1.0", "mf_230_1.0"};


        String[] folds = {"1", "2", "3", "4", "5"};
        //int[] groupSizes = {2, 3, 4, 5, 6, 7, 8};
        int[] groupSizes = {8};
        String[] groupTypes = {"div", "sim", "random"};

        for (String strategy : strategies) {
            System.out.println(strategy);
            for (int i = 0; i < DataFolders.length; i++) {
                String DATA_PATH = DataFolders[i];
                String fileName = individualRecFileName[i];
                for (int groupSize : groupSizes) {
                    System.out.println("group size: " + groupSize);
                    for (String groupType : groupTypes) {
                        System.out.println("group type: " + groupType);
                        String filePath = DATA_PATH + groupType + "_group_" + groupSize;


                        RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<Long, Long>(lp, lp);

                        Map<Long, List<Long>> groups = loadGroups(filePath);

                        for (String fold : folds) {
                            System.out.println("fold: " + fold);
                            String recIn = DATA_PATH + fold + "/" + fileName;
                            String out_file = DATA_PATH + fold + "/" + fileName + "_" + strategy.toLowerCase() + "_" + groupType + "_group_" + groupSize;

                            Map<Long, Recommendation<Long, Long>> recommendation = new HashMap<>();
                            format.getReader(recIn).readAll().forEach(rec -> {
                                recommendation.put(rec.getUser(), rec);
                            });
                            computeGroupRecs(out_file, strategy, groups, recommendation, format);

                        }
                    }
                }
            }
        }
    }

    public static void computeGroupRecs(String outFile, String strategy, Map<Long, List<Long>> groups,
                                        Map<Long, Recommendation<Long, Long>> recommendation,
                                        RecommendationFormat<Long, Long> format) {

        try {
            RecommendationFormat.Writer<Long, Long> writer = format.getWriter(outFile);
            groups.forEach((gID, members) -> {
                Recommendation<Long, Long> group_recs = null;
                switch (strategy) {
                    case "AVG":
                        group_recs = (new AverageScoreAggregationStrategy<Long, Long, Long>(gID, members, 10000)).aggregate(recommendation);
                        break;
                    case "FAI":
                        group_recs = (new FairnessAggregationStrategy<Long, Long, Long>(gID, members, 20)).aggregate(recommendation);
                        break;
                    case "XPO":
                        group_recs = (new XPO<Long, Long, Long>(gID, members, 20)).aggregate(recommendation);
                        break;
                    case "MAX":
                        group_recs = (new MaxSatisfactionAggStrategy<Long, Long, Long>(gID, members, 20).aggregate(recommendation));
                        break;
                }

                // Here write top-N recs for the group to a file!
                writer.accept(group_recs);
            });
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
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
