package com.distributed.mlp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.protocol.WeightSerializer;

public final class Inference {

    private static final String[] CIFAR10_LABELS = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    public static void main(String[] args) throws IOException {
        String weightsPath = args.length >= 1 ? args[0] : "results/model_weights.bin";
        int numSamples    = args.length >= 2 ? Integer.parseInt(args[1]) : 100;

        // Load weights
        System.out.println("Loading weights from " + weightsPath);
        byte[] bytes = Files.readAllBytes(Path.of(weightsPath));
        double[] weights = WeightSerializer.fromBytesDouble(bytes);

        // Restore model
        MLPModel model = new MLPModel();
        model.initXavier(42); // initialise structure, then overwrite
        model.loadWeights(weights);
        System.out.println("Model loaded. TOTAL_PARAMS=" + weights.length);

        // Load some test data (worker 0 of 1 = full dataset shard)
        System.out.println("Loading test data...");
        DataLoader loader = new DataLoader();
        List<Sample> samples = loader.loadShard(0, 30);
        int total = Math.min(numSamples, samples.size());

        // Run inference
        int correct = 0;
        for (int i = 0; i < total; i++) {
            Sample s = samples.get(i);
            double[] probs = model.forward(s.pixels());
            int predicted = argmax(probs);
            if (predicted == s.label()) correct++;

            if (i < 10) { // print first 10
                System.out.printf("Sample %3d | true=%-12s predicted=%-12s probs[true]=%.4f%n",
                        i,
                        CIFAR10_LABELS[s.label()],
                        CIFAR10_LABELS[predicted],
                        probs[s.label()]);
            }
        }

        double accuracy = 100.0 * correct / total;
        System.out.printf("%nAccuracy on %d samples: %d/%d = %.2f%%%n",
                total, correct, total, accuracy);
    }

    private static int argmax(double[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++)
            if (arr[i] > arr[best]) best = i;
        return best;
    }
}