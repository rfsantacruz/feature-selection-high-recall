package utils;

import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.DirectoryIteratorException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import problems.ClassificationProblem;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;


public class Util {


	public static List<ClassificationProblem> readAllFilesARFFFromDirectory(String path){
		List<ClassificationProblem> datasets = new ArrayList<ClassificationProblem>();

		Path dir = Paths.get(path);
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				String ext = com.google.common.io.Files.getFileExtension(file.toString());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION))
					datasets.add(new ClassificationProblem(file.toFile().getPath()));
			}
		} catch (IOException | DirectoryIteratorException x) {
			x.printStackTrace();
		}

		System.out.println(datasets.size() +" ARFF files were read");
		return datasets;
	}

	public static List<ClassificationProblem> readAllFilesARFFFromJar(String path){
		List<ClassificationProblem> dataSets = new ArrayList<ClassificationProblem>();

		Path dir = Paths.get(path);
		try {
			JarFile file = new JarFile(dir.toFile());
			Enumeration<JarEntry> e = file.entries();
			while(e.hasMoreElements()){
				JarEntry entry = e.nextElement();
				String ext = com.google.common.io.Files.getFileExtension(entry.getName());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION)){
					InputStreamReader inpr = new InputStreamReader(file.getInputStream(entry)) ;
					dataSets.add(new ClassificationProblem(entry.getName(),inpr));
				}

			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.println(dataSets.size() +" ARFF files were read");
		return dataSets;
	}


	//generate all possibles models based on param
	public static List<String> generateModels(List<Set<String>> paramsList){
		List<String> modelsStringSetting = null;
		final Joiner joiner = Joiner.on(" ").skipNulls();
		Function<List<String>, String> buildModelSettingString = new Function<List<String>, String>() {
			public String apply(List<String> params) {
				return joiner.join(params);
			}
		};

		if(paramsList != null && paramsList.size() > 1){
			Set<List<String>> cartesianProd = Sets.cartesianProduct(paramsList);

			if(cartesianProd != null){
				modelsStringSetting = Lists.newArrayList( Iterables.transform(cartesianProd, buildModelSettingString));
			}
		}

		return modelsStringSetting;
	}

	public static Set<String> generateModelsStringSettings(String settingCode, double... values){

		Set<String> ret = new HashSet<String>();
		for (double d : values) {
			ret.add(settingCode + " " + d);
		}
		return ret;
	}

	public static Logger getFileLogger(String logName, String logPath){
		try {
			SimpleFormatter formatter = new SimpleFormatter();  
			FileHandler file = new FileHandler(logPath);
			file.setFormatter(formatter);
			
			Logger.getLogger(logName).addHandler(file);
			
		} catch (SecurityException | IOException e) {
			e.printStackTrace();
		}
		
		return Logger.getLogger(logName);
	}
	
	public static String[] transformStringArray(int[] idxs){
		
		if(idxs == null | idxs.length <= 0)
			return null;
		
		String[] s = new String[idxs.length];
		for (int i = 0; i < s.length; i++) {
			s[i] = Integer.toString(idxs[i]);
		}
		return s;
	}
}
