package gfar;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import es.uam.eps.ir.ranksys.novdiv.reranking.Reranker;
import gfar.util.GFARPreferenceReader;
import org.jooq.lambda.Unchecked;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;
import gfar.rerank.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Supplier;
import static org.ranksys.formats.parsing.Parsers.lp;

public class RunGFAR {
    private static final String[] EXP_TYPE = {"EQUAL"};
    private static final String[] preName = {"GFAR"};

    public static void main(String[] args) throws Exception {
        String PROJECT_FOLDER = Paths.get(System.getProperty("user.dir")).getParent().toString();
        String[] DataFolders = {PROJECT_FOLDER + "/data/ml1m/",
                PROJECT_FOLDER + "/data/kgrec/"};
        String[] individualRecFileName = {"mf_30_1.0", "mf_230_1.0"};

        String[] folds = {"1", "2", "3", "4", "5"};

        String[] groupTypes = {"div", "sim", "random"};

        int[] groupSize = {8};
        //int[] groupSize = {2,3,4,5,6,7,8};


        double[] lambdas = {1.0};
        int cutoff = 10000;
        int maxLength = 20;

        for (int i = 0; i < DataFolders.length; i++) {
            String DATA_PATH = DataFolders[i];
            String individualRecsFileName = individualRecFileName[i];
            for (int exp_index = 0; exp_index < EXP_TYPE.length; exp_index++) {
                for (int size : groupSize) {
                    System.out.println("Group Size: " + size);
                    for (String groupType : groupTypes) {
                        System.out.println("Group Type: " + groupType);
                        String fileName = individualRecsFileName + "_avg_" + groupType + "_group_" + size;
                        String groupsFilePath = DATA_PATH + groupType + "_group_" + size;
                        Map<Long, List<Long>> groups = loadGroups(groupsFilePath);

                        for (String fold : folds) {
                            System.out.println("Fold: " + fold);
                            String recIn = DATA_PATH + fold + "/" + individualRecsFileName;
                            String trainDataPath = DATA_PATH + fold + "/" + "train.csv";
                            String groupRecIn = DATA_PATH + fold + "/" + fileName;
                            PreferenceData<Long, Long> trainingData = SimplePreferenceData.load(GFARPreferenceReader.
                                    get().read(trainDataPath, lp, lp));
                            RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<Long, Long>(lp, lp);
                            Map<Long, Recommendation<Long, Long>> individualRecommendations = new HashMap<>();

                            Map<String, Supplier<Reranker<Long, Long>>> rerankersMap = new HashMap<>();

                            System.out.println("Reading the individual score");
                            format.getReader(recIn).readAll().forEach(rec -> {
                                individualRecommendations.put(rec.getUser(), rec);
                            });
                            GroupIntentModel<Long, Long, Long> intentModel = new GFARIntentModel<>(groups, trainingData, EXP_TYPE[exp_index]);
                            GroupAspectModel<Long, Long, Long> aspectModel = new GFARAspectModel<>(maxLength, "BORDA", intentModel, individualRecommendations, groups);
                            for (double lambda : lambdas) {

                                rerankersMap.put(preName[exp_index] + "_" + lambda, () -> new GFAR<>(aspectModel, lambda, cutoff, true));
                            }

                            rerankersMap.forEach(Unchecked.biConsumer((name, rerankerSupplier) -> {
                                System.out.println("Running " + name);
                                String recOut = DATA_PATH + fold + "/" + fileName + "_" + name;
                                Reranker<Long, Long> reranker = rerankerSupplier.get();
                                try (RecommendationFormat.Writer<Long, Long> writer = format.getWriter(recOut)) {
                                    format.getReader(groupRecIn).readAll()
                                            .map(rec -> reranker.rerankRecommendation(new Recommendation<>(rec.getUser(), rec.getItems()), maxLength))
                                            .forEach(Unchecked.consumer(writer::write));
                                }

                            }));
                        }
                    }
                }
            }
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
