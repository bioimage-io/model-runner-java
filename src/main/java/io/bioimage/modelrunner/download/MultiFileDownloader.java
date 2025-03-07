/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.download;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

public class MultiFileDownloader {
	
	private final List<URL> urls;
	private final File folder;
	private final List<FileDownloader> downloaders;
	private final Thread thread;
	private final long totalSize;
	private AtomicLong progressSize = new AtomicLong(0);
	
	private Consumer<Long> totalProgress;
	
	private Consumer<Double> partialProgress;
	
	public MultiFileDownloader(List<URL> urls, File folder) {
		this(urls, folder, Thread.currentThread());
	}
	
	public MultiFileDownloader(List<URL> urls, File folder, Thread thread) {
		this.urls = urls;
		this.folder = folder;
		this.thread = thread;
		this.downloaders = new ArrayList<FileDownloader>();
		if (folder.exists() && folder.isFile())
			throw new IllegalArgumentException("A file with that name already exists. "
					+ "Please provide a valid folder name: " + folder.getAbsolutePath());
		if (!folder.isDirectory() && !folder.mkdirs())
			throw new RuntimeException("Unable to create folder in the path: " + folder.getAbsolutePath());
		long size = 0;
		for (URL url : urls) {
			try {
				String name = FileDownloader.getFileNameFromURLString(url.toString());
				FileDownloader fd = new FileDownloader(url.toString(), new File(folder, name), false);
				size += fd.getOnlineFileSize();
				downloaders.add(fd);
			} catch (MalformedURLException e) {
				e.printStackTrace();
			}
		}
		this.totalSize = size;
	}
	
	public File getFolder() {
		return this.folder;
	}
	
	public long getTotalSize() {
		return this.totalSize;
	}
	
	public long getDownloadedBytes() {
		return this.progressSize.get();
	}
	
	public void setTotalProgressConsumer(Consumer<Long> totalProgress) {
		this.totalProgress = totalProgress;
	}
	
	public void setPartialProgressConsumer(Consumer<Double> partialProgress) {
		this.partialProgress = partialProgress;
	}
	
	public void download() throws ExecutionException, InterruptedException {
        ExecutorService downloadExecutor = Executors.newFixedThreadPool(3);
        ScheduledExecutorService monitorExecutor = Executors.newScheduledThreadPool(1);

        List<Future<Void>> downloadFutures = new ArrayList<>();

        for (FileDownloader fd : downloaders) {
            downloadFutures.add(downloadExecutor.submit(() -> {
                fd.download(thread);
                return null;
            }));
        }

        monitorExecutor.scheduleAtFixedRate(
        		() -> monitorTotalProgress(downloadFutures), 0, 100, TimeUnit.MILLISECONDS);

        // Wait for all download tasks to complete
        try {
            for (Future<Void> future : downloadFutures)
                future.get();
        } finally {
            downloadExecutor.shutdown();
            monitorExecutor.shutdown();
        }
        finalProgress();
    }
	
	private void finalProgress() {
        AtomicLong total = new AtomicLong(0);
        for (FileDownloader fd : this.downloaders)
        	total.addAndGet(fd.getSizeDownloaded());
        if (totalProgress != null)
        	this.totalProgress.accept(total.get());
        if (partialProgress != null)
        	this.partialProgress.accept(total.get() / (double) this.totalSize);
        progressSize = total;
	}

    private void monitorTotalProgress(List<Future<Void>> downloadFutures) {
        if (!this.thread.isAlive()) {
            downloadFutures.stream().forEach(fut -> fut.cancel(true));
            return;
        }
        AtomicLong total = new AtomicLong(0);
        for (FileDownloader fd : this.downloaders)
        	total.addAndGet(fd.getSizeDownloaded());
        if (totalProgress != null)
        	this.totalProgress.accept(total.get());
        if (partialProgress != null)
        	this.partialProgress.accept(total.get() / (double) this.totalSize);
        progressSize = total;
    }
	
	/**
	 * Add the timestamp to the String given
	 * @param str
	 * 	String to add the time stamp
	 * @param isDir
	 * 	whether the file name represents a directory or not
	 * @return string with the timestamp
	 */
	public static String addTimeStampToFileName(String str, boolean isDir) {
		// Add timestamp to the model name. 
		// The format consists on: modelName + date as ddmmyyyy + time as hhmmss
        Calendar cal = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat("ddMMYYYY_HHmmss");
		String dateString = sdf.format(cal.getTime());
		if (isDir)
			return str + "_" + dateString;
		int ind = str.lastIndexOf(File.separator);
		String fileName = str;
		if (ind != -1)
			fileName = str.substring(ind + 1);
		int extensionPos = fileName.lastIndexOf(".");
		if (extensionPos == -1)
			return str + "_" + dateString;
		String nameNoExtension = str.substring(0, extensionPos);
		String extension = str.substring(extensionPos);
		return nameNoExtension + "_" + dateString + extension;
	}
}
