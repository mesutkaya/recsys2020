package gfar.rerank;

import es.uam.eps.ir.ranksys.core.Recommendation;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * GFAR's rel(u,i) based on the utility definition in "Fairness-Aware Group Recommendation with Pareto-Efficiency"
 * by Xiao et al. 2017.
 *
 * In our paper we use BORDA Relevance, but is possible to run experiments using BINARY Relevance as well.
 *
 * @param <G>
 * @param <I>
 * @param <U>
 */
public class GFARAspectModel<G, I, U> extends GroupAspectModel<G, I, U> {
    protected Map<G, List<U>> groupMembers;
    protected int maxLength;
    protected String RELTYPE;
    protected Map<U, Recommendation<U, I>> individualRecommendations;
    protected Map<U, Object2DoubleOpenHashMap<I>> relevances; //Binary or Borda relevance scores for individuals, based on their individual top-N's.


    public GFARAspectModel(int maxLength, String RELTYPE, GroupIntentModel<G, I, U> intentModel,
                           Map<U, Recommendation<U, I>> individualRecommendations,
                           Map<G, List<U>> groupMembers) {
        super(intentModel);
        this.groupMembers = groupMembers;
        this.maxLength = maxLength;
        this.individualRecommendations = individualRecommendations;
        this.RELTYPE = RELTYPE;
        relevances = new HashMap<>();

        individualRecommendations.forEach((u, uiRecommendation) -> {
            Object2DoubleOpenHashMap<I> iRelevances = new Object2DoubleOpenHashMap<>();
            AtomicInteger rank = new AtomicInteger(1);
            uiRecommendation.getItems().stream().limit(maxLength).forEach(iTuple2od -> {
                double binRel = 1.0;
                double bordaRel = maxLength - rank.get();
                rank.getAndIncrement();


                if (RELTYPE.equals("BINARY"))
                    iRelevances.addTo(iTuple2od.v1, binRel);
                else if (RELTYPE.equals("BORDA"))
                    iRelevances.addTo(iTuple2od.v1, bordaRel);

            });
            relevances.put(u, iRelevances);
        });

    }

    @Override
    protected GroupMembersAspectModel get(G g) {
        return new RelevanceGroupAspectModel(g);
    }

    public class RelevanceGroupAspectModel extends GroupMembersAspectModel {
        protected G group;

        public RelevanceGroupAspectModel(G g) {
            super(g);
            this.group = g;
        }

        @Override
        public ItemAspectModel<I, U> getItemAspectModel(List<Tuple2od<I>> items) {
            Object2DoubleOpenHashMap<U> probNorm = new Object2DoubleOpenHashMap<>();
            items.forEach(iTuple2od -> {
                groupMembers.get(group).forEach(u -> {
                    if (individualRecommendations.containsKey(u)) {
                        AtomicInteger rank = new AtomicInteger(1);
                        individualRecommendations.get(u).getItems().stream().limit(maxLength).forEach(iTuple2od1 -> {
                            double binRel = 1.0;
                            double bordaRel = maxLength - rank.get();
                            rank.getAndIncrement();
                            if (RELTYPE.equals("BINARY"))
                                probNorm.addTo(u, binRel);
                            else if (RELTYPE.equals("BORDA"))
                                probNorm.addTo(u, bordaRel);
                        });
                    }
                });
            });
            return (iv, user) -> individualRecommendations.containsKey(user) && relevances.get(user).containsKey(iv.v1) ? relevances.get(user).get(iv.v1) / probNorm.getDouble(user) : 0.0;
        }

    }
}
