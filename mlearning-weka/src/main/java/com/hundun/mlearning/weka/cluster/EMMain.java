package com.hundun.mlearning.weka.cluster;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;

/**
 * @DESC cluster model : EM
 * @author xinshiyou
 */
public class EMMain {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		// Weka instance
		Instances instance = new Instances(new BufferedReader(new FileReader(args[0])));

		// EM model
		EM model = new EM();
		model.buildClusterer(instance);

		System.out.println("Model:\n" + model);

		// measure model
		double logLikelihood = ClusterEvaluation.crossValidateModel(model, instance, 10, new Random(1));
		System.out.println("log likelyhood: " + logLikelihood);

	}

}
