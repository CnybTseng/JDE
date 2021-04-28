/******************************************************************************

                  版权所有 (C), 2004-2020, 成都思晗科技股份有限公司

 ******************************************************************************
  文 件 名   : test.java
  版 本 号   : 初稿
  作    者   : Zeng Zhiwei
  生成日期   : 2020年9月24日
  最近修改   :
  功能描述   : 多目标跟踪的JNI接口的测试模块
  
  修改历史   :
  1.日    期   : 2020年9月24日
    作    者   : Zeng Zhiwei
    修改内容   : 创建文件

******************************************************************************/

import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;
import java.util.Collections;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import org.json.JSONArray;
import org.json.JSONObject;
import com.sihan.system.jni.utils.mot4j;
import com.sihan.system.jni.utils.CrossLineDetector;

public class test
{
    public static void main(String argv[])
    {
        if (argv.length < 2)
        {
            System.err.printf("Usage:\n\tjava test CONFIG" +
                " IMAGE_DIR[ RESET=1[ ITERS=1]]\n");
            return;
        }
        test t = new test(argv);
    }
    public test(String argv[])
    {        
        int reset = 1;  // 是否定期重置轨迹(轨迹分段)
        if (argv.length >= 3)
        {
            reset = Integer.parseInt(argv[2]);
        }
        
        int iters = 1;  // 循环测试次数, 以较短时间的图像序列模拟长时间跟踪
        if (argv.length >= 4)
        {
            iters = Integer.parseInt(argv[3]);
        }
        
        // 1. 加载模型
        mot4j handle = new mot4j();
        if (0 != handle.load_mot_model(argv[0]))
        {
            System.err.println("load_mot_model fail\n");
            return;
        }
        
        int channel = 20;
        CrossLineDetector cld = new CrossLineDetector();
        if (!cld.set_line(channel, 61, 867, 319, 842, 139, 1079))
        {
            System.err.println("set_line fail\n");
        }
        
        // 读取文件夹下的图片序列, 图片名必须按照名字排序, 保证时间先后顺序不乱
        File file = new File(argv[1]);
        File fs[] = file.listFiles();
        Arrays.sort(fs);
        int timestamp = 0;
        List<File> files = Arrays.asList(fs);
        for (int iter = 0; iter < iters; iter++)
        {
            for (File f : files)
            {
                if (f.isDirectory())
                {
                    continue;
                }

                // 读取图片并解码
                System.out.println(f);                
                BufferedImage im;
                try
                {
                    im = ImageIO.read(f);
                }
                catch (Exception e)
                {
                    System.err.printf("ImageIO.read %f fail\n", f);
                    continue;
                }

                int stride = im.getWidth() * 3;
                // 应将data由RGB888转BGR888, 此处仅用于展示算法API调用方法, 就不转了
                byte data[] = ((DataBufferByte)im.getRaster().getDataBuffer()).getData();
                
                for (int i = 0; i < data.length; i += 3)
                {
                    byte swap = data[i];
                    data[i] = data[i + 2];
                    data[i + 2] = swap;
                }
                
                // 2. 执行模型推理
                String result = handle.forward_mot_model(data, im.getWidth(), im.getHeight(), stride, channel);
                if (iters > 1)
                {
                    continue;
                }
                
                // 循环测试不打印下面的信息
                // 拿着跟踪结果去绘图, 或抠图, 或啥的......
                // System.out.println(result);                
                try
                {                
                    JSONArray array = new JSONArray(result);                
                    for (int j = 0; j < array.length(); j++)
                    {
                        JSONObject jobj = array.getJSONObject(j);
                        int identifier = jobj.getInt("identifier");     // 目标轨迹ID
                        String category = jobj.getString("category");   // 目标类别
                        JSONArray rects = jobj.getJSONArray("rects");   // 目标边框集合
                        // System.out.printf("%d, %s:\n", identifier, category);
                        for (int k = 0; k < rects.length(); k++)
                        {
                            JSONObject kobj = rects.getJSONObject(k);
                            int x = kobj.getInt("x");
                            int y = kobj.getInt("y");
                            int w = kobj.getInt("width");
                            int h = kobj.getInt("height");
                            // System.out.printf("\t%d %d %d %d\n", x, y, w, h);
                        }
                    }
                    
                    if (array.length() > 0)
                    {
                        int recall = 2;
                        String event = cld.detect_cross_event(result, recall);
                        System.out.printf(event);
                        
                        String filename = String.format("./track/%06d.json", timestamp);
                        try (FileWriter writer = new FileWriter(filename))
                        {
                            writer.write(result);
                        }
                        catch (Exception e)
                        {
                            System.err.println("store tracks fail\n");
                        }
                        
                        String totals = handle.get_total_tracks(reset, channel);
                        try
                        {
                            JSONArray array2 = new JSONArray(totals);
                            String filename2 = String.format("./tracklet/%06d.json", timestamp);
                            try (FileWriter writer = new FileWriter(filename2))
                            {
                                writer.write(totals);
                            }
                            catch (Exception e)
                            {
                                System.err.println("store tracklet fail\n");
                            }
                        }
                        catch (Exception e)
                        {
                            ;
                        }
                    }
                }
                catch (Exception e)
                {
                    System.err.println("Empty result");
                }
                timestamp++;
            }
            Collections.reverse(files); // 逆序排列图像文件, 轨迹将会平滑过渡
        }
        
        // 3. 卸载模型
        handle.unload_mot_model();
    }
}