import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

public class KNNAlgorithm {

    private final int testingSetSize = 20;

    private List<Iris> testingSet;

    private List<Iris> dataset;

    private int k;

    private KNNAlgorithm(int k) {
        this.k = k;
        Reader reader = new Reader();
        dataset = new ArrayList<>();
        dataset.addAll(reader.dataset);

        testingSet = new ArrayList<>();
        selectTestingSet();
        dataset.removeAll(testingSet);

        kNNAlgorithm();
    }

    public static void main(String[] args) {
        new KNNAlgorithm(new Scanner(System.in).nextInt());
    }

    private void kNNAlgorithm() {
        for (int i = 0; i < testingSetSize; i++) {
            Iris[] neighbors = findKNearestNeighbors(testingSet.get(i));
            testingSet.get(i).predictedType = predict(neighbors);
        }
        calcAccuracy();
    }

    private void calcAccuracy() {
        int correctPrediction = 0;
        for (int i = 0; i < testingSetSize; i++) {
            if (testingSet.get(i).predictedType == testingSet.get(i).type)
                correctPrediction++;
        }

        for (int i = 0; i < testingSetSize; i++) {
            System.out.println(testingSet.get(i) + " ---but predicted---> "
                                               + testingSet.get(i).printType(testingSet.get(i).predictedType));
        }
        System.out.println("The accuracy is " + ((double) correctPrediction / testingSetSize) * 100 + "%");
    }

    private Iris[] findKNearestNeighbors(Iris record) {
        Iris[] neighbors = new Iris[k];

        //the first k dataset
        int index;
        for (index = 0; index < k; index++) {
            dataset.get(index).distance = getDistance(dataset.get(index), record);
            neighbors[index] = dataset.get(index);
        }

        //go through the remaining records in the trainingSet to find K nearest neighbors
        for (index = k; index < dataset.size(); index++) {
            dataset.get(index).distance = getDistance(dataset.get(index), record);

            //get the index of the neighbor with the largest distance to testRecord
            int maxIndex = 0;
            for (int i = 1; i < k; i++) {
                if (neighbors[i].distance > neighbors[maxIndex].distance)
                    maxIndex = i;
            }

            //add the current trainingSet[index] into neighbors if applicable
            if (neighbors[maxIndex].distance > dataset.get(index).distance)
                neighbors[maxIndex] = dataset.get(index);
        }

        return neighbors;
    }

    private double getDistance(Iris train, Iris test) {
        double sum2;
        sum2 = Math.pow(train.sepalLength - test.sepalLength, 2);
        sum2 += Math.pow(train.sepalWidth - test.sepalWidth, 2);
        sum2 += Math.pow(train.petalLength - test.petalLength, 2);
        sum2 += Math.pow(train.petalWidth - test.petalWidth, 2);
        return Math.sqrt(sum2);
    }

    private int predict(Iris[] neighbors) {
        //construct a HashMap to store <type, distance>
        Map<Integer, Double> map = new HashMap<>();
        for (int index = 0; index < k; index++) {
            Iris temp = neighbors[index];
            int key = temp.type;

            //if this classLabel does not exist in the HashMap, put <key, 1/(temp.distance)> into the HashMap
            if (!map.containsKey(key))
                map.put(key, 1 / temp.distance);
                //else, update the HashMap by adding the distance associating with that key
            else {
                double value = map.get(key);
                value += 1 / temp.distance;
                map.put(key, value);
            }
        }

        //Find the most likely type
        double maxSimilarity = 0;
        int returnType = -1;
        Set<Integer> typeSet = map.keySet();

        //go through the HashMap by using keys
        //and find the key with the highest weights
        for (Integer type : typeSet) {
            double value = map.get(type);
            if (value > maxSimilarity) {
                maxSimilarity = value;
                returnType = type;
            }
        }
        return returnType;
    }

    private void selectTestingSet() {
        int i = 0;
        boolean[] testingIndexes = new boolean[150];
        while (i < testingSetSize) {
            int random = ThreadLocalRandom.current().nextInt(0, 150);
            if (!testingIndexes[random]) {
                testingIndexes[random] = true;
                i++;
                testingSet.add(dataset.get(random));
            }
        }
    }

    class Iris {
        double sepalLength;

        double sepalWidth;

        double petalLength;

        double petalWidth;

        int type;

        int predictedType;

        double distance;

        Iris(double sepalLength, double sepalWidth, double petalLength, double petalWidth, int type) {
            distance = 0;
            this.sepalLength = sepalLength;
            this.sepalWidth = sepalWidth;
            this.petalLength = petalLength;
            this.petalWidth = petalWidth;
            this.type = type;
        }

        String printType(int type) {
            if (type == 1) {
                return "setosa";
            } else if (type == 2) {
                return "versicolour";
            } else
                return "virginica";
        }

        @Override public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(sepalLength);
            sb.append(",");
            sb.append(sepalWidth);
            sb.append(",");
            sb.append(petalLength);
            sb.append(",");
            sb.append(petalWidth);
            sb.append(",");
            sb.append(printType(type));
            return sb.toString();
        }
    }

    class Reader {
        List<Iris> dataset;

        Reader() {
            dataset = new ArrayList<>();
            String line;
            try (FileReader input = new FileReader("iris.txt");
                 BufferedReader br = new BufferedReader(input)
            ) {
                int i = 0;
                while ((line = br.readLine()) != null) {

                    String[] data = line.split(",");
                    if (i < 50) {
                        dataset.add(new Iris(Double.parseDouble(data[0]), Double.parseDouble(data[1]),
                                             Double.parseDouble(data[2]), Double.parseDouble(data[3]), 1));
                    } else if (i < 100) {
                        dataset.add(new Iris(Double.parseDouble(data[0]), Double.parseDouble(data[1]),
                                             Double.parseDouble(data[2]), Double.parseDouble(data[3]), 2));
                    } else {
                        dataset.add(new Iris(Double.parseDouble(data[0]), Double.parseDouble(data[1]),
                                             Double.parseDouble(data[2]), Double.parseDouble(data[3]), 3));
                    }
                    i++;
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
