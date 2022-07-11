using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Net;
using PacketDotNet;
using SharpPcap;
using SharpPcap.LibPcap;

namespace Bupt;

/// <summary>
/// 检测音视频块
/// </summary>
public static class ChunkDetection {
	/// <summary>
	/// 代表一个GET请求
	/// </summary>
	public class GetRequest {
		/// <summary>
		/// 排序
		/// </summary>
		public int Index { get; set; }
		/// <summary>
		/// 请求的目标地址
		/// </summary>
		public IPAddress Dst { get; }

		/// <summary>
		/// 发送GET请求的时间戳
		/// </summary>
		public PosixTimeval DownStart { get; }

		public ProtocolType ProtocolType { get; }

		/// <summary>
		/// 第一个下载的包
		/// </summary>
		public IPPacket? FirstDownPacket { get; set; }

		public IPPacket? LastDownPacket { get; set; }

		public PosixTimeval? DownEnd { get; set; }

		public int PackageCount { get; set; }

		public int TotalSize { get; set; }

		public double AvgPackageSize => (double)TotalSize / PackageCount;

		public GetRequest(IPAddress dst, PosixTimeval downStart, ProtocolType protocolType) {
			Dst = dst;
			DownStart = downStart;
			ProtocolType = protocolType;
		}
	}

	private static void WriteRequestsToCsv(IReadOnlyCollection<GetRequest> requests, string savePath) {
		if (requests.Count == 0) {
			return;
		}
		using var sw = new StreamWriter(new FileStream(savePath, FileMode.Create));
		sw.WriteLine("index,pkg_count,avg_pkg_size,down_time,down_start");
		foreach (var req in requests) {
			sw.WriteLine($"{req.Index},{req.PackageCount},{req.AvgPackageSize},{(req.DownEnd!.Date - req.DownStart.Date).TotalSeconds},{((DateTimeOffset)req.DownStart.Date).ToUnixTimeMilliseconds()}");
		}
	}

	public static void Analyze(string pcapPath) {
		var device = new CaptureFileReaderDevice(pcapPath);
		device.Open();

		var allRequests = new List<GetRequest>();

		// key为目标ip
		var getRequests = new Dictionary<IPAddress, GetRequest>();

		while (device.GetNextPacket(out var packet) > 0) {
			if (Packet.ParsePacket(LinkLayers.Ethernet, packet.Data.ToArray()) is not EthernetPacket ethPacket) {
				continue;
			}
			var ipPacket = ethPacket.Extract<IPPacket>();
			if (ipPacket == null) {
				continue;
			}
			// GET threshold
			if (ipPacket.PayloadLength < 300) {
				continue;
			}
			var dst = ipPacket.DestinationAddress;
			var src = ipPacket.SourceAddress;
			if (src.ToString().StartsWith("192.168")) {
				// 原地址是192.168.xxx.xxx，说明是GET请求
				if (getRequests.TryGetValue(dst, out var getRequest)) {
					// 说明这个曾经GET过，这是一个新的GET，那就把旧的保存下来
					if (getRequest.TotalSize > 80 * 1024) {  // 只有当大于80KB才视为有效
						getRequest.Index = allRequests.Count;
						allRequests.Add(getRequest);
					}
				}
				getRequests[dst] = new GetRequest(dst, packet.Header.Timeval, ipPacket.Protocol);
			} else if (getRequests.TryGetValue(src, out var getRequest)) {
				// 否则如果曾经请求过，那就说明这是服务器的回传，即下载的包
				getRequest.FirstDownPacket ??= ipPacket;  // 第一个包为空才设置
				getRequest.LastDownPacket = ipPacket;
				getRequest.DownEnd = packet.Header.Timeval;
				getRequest.PackageCount++;
				getRequest.TotalSize += ipPacket.PayloadLength;
			}
		}

		if (allRequests.Count == 0) {
			return;
		}

		// 处理完成之后，要根据平均大小筛选音视频包
		var filePathWithoutExt = Path.Combine(Path.GetDirectoryName(pcapPath)!, Path.GetFileNameWithoutExtension(pcapPath));
		var videoRequests = new List<GetRequest>();
		var audioRequests = new List<GetRequest>();
		for (var i = 0; i < allRequests.Count; i += 10) {
			var packages = allRequests.Skip(i).Take(10).ToImmutableArray();
			var avgSize = packages.Average(p => p.AvgPackageSize);
			videoRequests.AddRange(packages.Where(p => p.AvgPackageSize >= avgSize));
			audioRequests.AddRange(packages.Where(p => p.AvgPackageSize < avgSize));
		}
		WriteRequestsToCsv(videoRequests, filePathWithoutExt + "_videos.csv");
		WriteRequestsToCsv(audioRequests, filePathWithoutExt + "_audios.csv");
	}
}