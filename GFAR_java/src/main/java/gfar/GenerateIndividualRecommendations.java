package gfar;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import es.uam.eps.ir.ranksys.mf.Factorization;
import es.uam.eps.ir.ranksys.mf.rec.MFRecommender;
import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.ranksys.rec.runner.RecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilterRecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilters;
import gfar.util.GFARPreferenceReader;
import mf.MFFactorizer;
import org.jooq.lambda.Unchecked;
import org.ranksys.formats.index.ItemsReader;
import org.ranksys.formats.index.UsersReader;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import static org.ranksys.formats.parsing.Parsers.lp;
import java.nio.file.Paths;

/**
 * Using the hyper-parameter values for the datasets we train the MF on
 * the union of the training and validation sets, which we will refer to as 𝑅_{train}. Once trained, we can obtain s(𝑢, 𝑖)
 * for all users 𝑢 ∈ 𝑈 and items 𝑖 ∈ 𝐼 . Specifically, if 𝑟𝑢𝑖 ∉ 𝑅train (an unseen item), we use the MF model to predict
 * s(𝑢, 𝑖). Once we have scores, 𝑠 (𝑢, 𝑖), it is possible to compute individual top-𝑁𝑢 and 𝑝(𝑟𝑒𝑙 |𝑢, 𝑖) for those algorithms
 * that need them.
 */
public class GenerateIndividualRecommendations {

    public static void main(String[] args) throws IOException {
        String PROJECT_FOLDER = Paths.get(System.getProperty("user.dir")).getParent().toString();
        String[] DataFolders = {PROJECT_FOLDER + "/data/ml1m/",
                PROJECT_FOLDER + "/data/kgrec/"};
        // The optimized hyper-parameter values for the datasets!
        int[] latentFactors = {30, 230};
        double[] alphaValues = {1.0, 1.0};

        for (int i = 0; i < DataFolders.length; i++) {
            String DataFolder = DataFolders[i];
            int numOfLatentFactors = latentFactors[i];
            double alpha = alphaValues[i];
            String userPath = DataFolder + "/users.txt";
            String itemPath = DataFolder + "/items.txt";

            FastUserIndex<Long> userIndex = SimpleFastUserIndex.load(UsersReader.read(userPath, lp));
            FastItemIndex<Long> itemIndex = SimpleFastItemIndex.load(ItemsReader.read(itemPath, lp));
            String[] folds = {"1", "2", "3", "4", "5"};
            for (String fold : folds) {
                String trainDataPath = DataFolder + "/" + fold + "/train.csv";
                String testDataPath = DataFolder + "/" + fold + "/test.csv";
                FastPreferenceData<Long, Long> trainData = SimpleFastPreferenceData.load(GFARPreferenceReader
                        .get().read(trainDataPath, lp, lp), userIndex, itemIndex);
                FastPreferenceData<Long, Long> testData = SimpleFastPreferenceData.load(GFARPreferenceReader
                        .get().read(testDataPath, lp, lp), userIndex, itemIndex);

                //////////////////
                // MF RECOMMENDER //
                //////////////////
                Map<String, Supplier<Recommender<Long, Long>>> recMap = new HashMap<>();

                // Matrix factorization of Pilaszy et al. 2010
                recMap.put(DataFolder + "/" + fold + "/mf_" + numOfLatentFactors + "_" + alpha, () -> {
                    double lambda = 0.1;
                    DoubleUnaryOperator confidence = x -> 1 + alpha * x;
                    int numIter = 100;
                    Factorization<Long, Long> factorization = new MFFactorizer<Long, Long>(lambda, confidence, numIter).factorize(numOfLatentFactors, trainData);
                    return new MFRecommender<>(userIndex, itemIndex, factorization);
                });


                ////////////////////////////////
                // GENERATING RECOMMENDATIONS //
                ////////////////////////////////
                Set<Long> targetUsers = testData.getUsersWithPreferences().collect(Collectors.toSet());
                System.out.println(targetUsers.size());

                //Set<Long> targetUsers = loadTestUsers(groupUsersPath);
                RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<>(lp, lp);
                Function<Long, IntPredicate> filter = FastFilters.notInTrain(trainData);
                // For both ML1M and KGRec-Music datasets, total number of items are less than 10,000 so we set to this number
                int maxLength = 10000;
                RecommenderRunner<Long, Long> runner = new FastFilterRecommenderRunner<>(userIndex, itemIndex, targetUsers.stream(), filter, maxLength);

                recMap.forEach(Unchecked.biConsumer((name, recommender) -> {
                    System.out.println("Running " + name);
                    try (RecommendationFormat.Writer<Long, Long> writer = format.getWriter(name)) {
                        runner.run(recommender.get(), writer);
                    }
                }));
            }
        }
    }

    /**
     * If we want to generate recommendations for only a subset of users, we should create a file that each line
     * consists of user IDs, and use this method.
     *
     * @param filePath
     * @return
     */
    public static Set<Long> loadTestUsers(String filePath) {
        Scanner s = null;
        try {
            s = new Scanner(new File(filePath));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        Set<Long> list = new HashSet<>();
        assert s != null;
        while (s.hasNext()) {
            list.add(Long.parseLong(s.next()));
        }
        s.close();
        return list;
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
