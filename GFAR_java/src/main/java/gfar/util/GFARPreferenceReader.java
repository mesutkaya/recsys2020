package gfar.util;

import org.jooq.lambda.tuple.Tuple;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.preference.PreferencesReader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.stream.Stream;

import static es.uam.eps.ir.ranksys.core.util.FastStringSplitter.split;
import static java.lang.Double.parseDouble;

public class GFARPreferenceReader implements PreferencesReader {
    public static <U, I> GFARPreferenceReader get() {
        return new GFARPreferenceReader();
    }

    @Override
    public <U, I> Stream<Tuple3<U, I, Double>> read(InputStream in, Parser<U> up, Parser<I> ip) throws IOException {
        return new BufferedReader(new InputStreamReader(in)).lines().map(line -> {
            CharSequence[] tokens = split(line, ',', 4);
            U user = up.parse(tokens[0]);
            I item = ip.parse(tokens[1]);
            double value = parseDouble(tokens[2].toString());


            return Tuple.tuple(user, item, value);
        });
    }
}
