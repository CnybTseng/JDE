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
import java.util.Arrays;
import java.util.Vector;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import org.json.JSONArray;
import org.json.JSONObject;

public class test
{
    public static void main(String argv[])
    {
        if (argv.length < 2)
        {
            System.err.printf("Usage:\n\tjava test /path/to/.yaml");
            System.err.printf(" /path/to/images [max_frames, default is 5]\n");
            return;
        }
        test t = new test(argv);
    }
    public test(String argv[])
    {
        // 当有一大堆测试图片的时候, 您也许只想测试前面几张
        int max_frames = 5;
        if (argv.length >= 3)
        {
            max_frames = Integer.parseInt(argv[2]);
        }
        
        // 1. 加载模型
        mot4j handle = new mot4j();
        if (0 != handle.load_mot_model(argv[0]))
        {
            System.err.println("load_mot_model fail\n");
            return;
        }
        
        // 读取文件夹下的图片序列, 图片名必须按照名字排序, 保证时间先后顺序不乱
        File file = new File(argv[1]);
        File fs[] = file.listFiles();
        Arrays.sort(fs);

        for (File f : fs)
        {
            if (f.isDirectory())
            {
                continue;
            }
            
            if (max_frames <= 0)
            {
                break;
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
            
            max_frames--;
            int stride = im.getWidth() * 3;
            // 应将data由RGB888转BGR888, 此处仅用于展示算法API调用方法, 就不转了
            byte data[] = ((DataBufferByte)im.getRaster().getDataBuffer()).getData();
            
            // 2. 执行模型推理
            String result = handle.forward_mot_model(data, im.getWidth(), im.getHeight(), stride);
            
            // 拿着跟踪结果去绘图, 或抠图, 或啥的......
            System.out.println(result);                
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
                        if (x == 0 && y == 0 && w == 0 && h == 0)
                        {
                            continue;
                        }
                        // System.out.printf("\t%d %d %d %d\n", x, y, w, h);
                    }
                }
            }
            catch (Exception e)
            {
                System.err.println("Empty result");
            }            
        }
        
        // 3. 卸载模型
        handle.unload_mot_model();
    }
}