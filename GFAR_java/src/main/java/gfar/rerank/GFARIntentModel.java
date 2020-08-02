package gfar.rerank;

import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Fairness-Aware Group Recommendation, Adapting Intent-aware Recommenders for diversification
 * to the group recommendations scenario!
 *
 * @param <G>
 * @param <I>
 * @param <U>
 */
public class GFARIntentModel<G, I, U> extends GroupIntentModel<G, I, U> {
    protected Map<G, List<U>> group_members;
    protected PreferenceData<U, I> trainingData;
    protected String EXP_TYPE;

    public GFARIntentModel(Map<G, List<U>> group_members, PreferenceData<U, I> trainingData, String EXP_TYPE) {
        this.group_members = group_members;
        this.trainingData = trainingData;
        this.EXP_TYPE = EXP_TYPE; // {SIZE, INV_SIZE, EQUAL}
    }

    @Override
    protected GroupMembersIntentModel<G, I, U> get(G group) {
        return new SPGroupIntentModel(group);
    }

    public class SPGroupIntentModel implements GroupMembersIntentModel<G, I, U> {
        protected final Object2DoubleOpenHashMap<U> pug;
        protected final List<U> members;
        protected G group;

        public SPGroupIntentModel(G group) {
            this.group = group;
            this.members = group_members.get(group);
            int group_size = members.size();
            int profileSizes = 0;
            pug = new Object2DoubleOpenHashMap<>();
            //  fill pug here! Either they are equal or based on size!


            if(EXP_TYPE.equals("EQUAL")) {
                for (U u : members) {

                    pug.put(u, 1 / (double) group_size);
                }
            }
            else{
                            for (U u : members) {
                if (trainingData.containsUser(u))
                    profileSizes += trainingData.numItems(u);
            }
            for (U u : members) {
                if (trainingData.containsUser(u)) {
                    if(EXP_TYPE.equals("INV_SIZE")) pug.put(u, (profileSizes - trainingData.numItems(u)) / (double) profileSizes);
                    if(EXP_TYPE.equals("SIZE")) pug.put(u, (trainingData.numItems(u)) / (double) profileSizes);
                }
            }
            }
        }

        @Override
        public Set<U> getIntents() {
            return pug.keySet();
        }

        @Override
        public Stream<U> getItemIntents(I i) {
            // TODO return item intents, namely which group members has a predicted rating for the item i! Check xQuAD!
            return members.stream();
        }

        @Override
        public double pu_g(U u) {
            return pug.getDouble(u);
        }
    }
}
