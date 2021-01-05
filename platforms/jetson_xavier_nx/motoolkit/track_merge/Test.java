import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.nio.file.Files;
import com.sihan.system.jni.utils.TrackMerge;

public class Test
{
    public static void main(String argv[])
    {
        Test test = new Test(argv);
    }
    public Test(String argv[])
    {
        TrackMerge handle = new TrackMerge();
        String registered = handle.get_registered_channels();
        System.out.printf(registered);

        // Load tracks from the first channel.
        String tracks1 = new String();
        try
        {
            StringBuffer buffer = new StringBuffer();
            BufferedReader reader = new BufferedReader(new FileReader(argv[0]));
            String s = null;
            while((s = reader.readLine()) != null)
            {
                buffer.append(s.trim());
            }
            tracks1 = buffer.toString();
        }
        catch (Exception e)
        {
            System.err.println("load tracks1 fail\n");
        }
        
        // Load tracks from the second channel.
        String tracks2 = new String();
        try
        {
            StringBuffer buffer2 = new StringBuffer();
            BufferedReader reader2 = new BufferedReader(new FileReader(argv[1]));
            String s = null;
            while((s = reader2.readLine()) != null)
            {
                buffer2.append(s.trim());
            }
            tracks2 = buffer2.toString();
        }
        catch (Exception e)
        {
            System.err.println("load tracks2 fail\n");
        }
        
        // System.out.println(tracks1);
        // System.out.println(tracks2);
        
        // Merge tracks and save results.
        int channel1 = Integer.parseInt(argv[2]);
        int channel2 = Integer.parseInt(argv[3]);
        int cost_thresh = Integer.parseInt(argv[4]);
        String tracks = handle.merge_track(tracks1, tracks2, channel1, channel2, cost_thresh);
        try (FileWriter writer = new FileWriter("tracks.json"))
        {
            writer.write(tracks);
        }
        catch (Exception e)
        {
            System.err.println("store tracks fail\n");
        }
    }
}