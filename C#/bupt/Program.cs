namespace Bupt;

public static class Program {
	private static void Main(string[] args) {
		foreach (var pcapPath in Directory.EnumerateFiles(@"G:\Source\Python\AI\bupt\data\A\PCAP_FILES", "*.pcap")) {
			ChunkDetection.Analyze(pcapPath);
		}
	}
}