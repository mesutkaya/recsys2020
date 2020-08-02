package gfar.rerank;

import es.uam.eps.ir.ranksys.core.model.UserModel;
import es.uam.eps.ir.ranksys.diversity.intentaware.IntentModel;

import java.util.Set;
import java.util.stream.Stream;

public abstract class GroupIntentModel<G,I,U> extends UserModel<G> {

    public GroupIntentModel(Stream<G> targetGroups){super(targetGroups);}
    public GroupIntentModel(){super();}

    @Override
    protected abstract GroupIntentModel.GroupMembersIntentModel<G, I, U> get(G group);

    @SuppressWarnings("unchecked")
    @Override
    public GroupIntentModel.GroupMembersIntentModel<G, I, U> getModel(G group) {
        return (GroupIntentModel.GroupMembersIntentModel<G, I, U>) super.getModel(group);
    }

    /**
     * Group members intent-aware model for {@link IntentModel}.
     * @param <G> group type
     * @param <I> item type
     * @param <U> user type
     */
    public interface GroupMembersIntentModel<G, I, U> extends Model<G> {

        /**
         * Returns the intents considered in the intent model, every group member in this case is an intent.
         *
         * @return the intents considered in the intent model
         */
        public abstract Set<U> getIntents();

        /**
         * Returns the intents associated with an item.
         *
         * @param i item
         * @return the intents associated with the item
         */
        public abstract Stream<U> getItemIntents(I i);

        /**
         * Returns the probability of an intent in the model.
         *
         * @param u intent
         * @return probability of an intent in the model
         */
        public abstract double pu_g(U u);
    }
}
