import argparse
from src.paper_manager import add_and_classify_papers, search_papers, organize_existing_folder
from src.image_manager import index_images, search_images


def main():
    parser = argparse.ArgumentParser(description="本地 AI 文献与图像管理助手")
    subparsers = parser.add_subparsers(dest="command")

    add_paper_parser = subparsers.add_parser("add_paper", help="添加/索引论文 PDF（自动归类并归档）")
    add_paper_parser.add_argument("path", type=str, help="PDF 文件或文件夹路径")
    add_paper_parser.add_argument("--topics", type=str, default="", help='主题列表，例如 "CV,NLP,RL"')
    add_paper_parser.add_argument("--copy", action="store_true", help="不移动原文件，改为复制到归档目录")

    organize_parser = subparsers.add_parser("organize_papers", help="一键整理现有混乱文件夹（自动归类归档）")
    organize_parser.add_argument("path", type=str, help="要整理的文件夹路径")
    organize_parser.add_argument("--topics", type=str, default="", help='主题列表，例如 "CV,NLP,RL"')
    organize_parser.add_argument("--copy", action="store_true", help="不移动原文件，改为复制到归档目录")

    search_paper_parser = subparsers.add_parser("search_paper", help="语义搜索论文")
    search_paper_parser.add_argument("query", type=str, help="自然语言查询")
    search_paper_parser.add_argument("--top_k", type=int, default=5)
    search_paper_parser.add_argument("--topic", type=str, default="", help="只在某个主题下搜索（可选）")

    index_img_parser = subparsers.add_parser("index_image", help="索引图片文件夹")
    index_img_parser.add_argument("path", type=str, help="图片文件或文件夹路径")

    search_img_parser = subparsers.add_parser("search_image", help="以文搜图")
    search_img_parser.add_argument("query", type=str, help="自然语言描述")
    search_img_parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "add_paper":
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        add_and_classify_papers(args.path, topics, move_to_topic_dir=(not args.copy))

    elif args.command == "organize_papers":
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        organize_existing_folder(args.path, topics, move_to_topic_dir=(not args.copy))

    elif args.command == "search_paper":
        topic = args.topic.strip() or None
        search_papers(args.query, top_k=args.top_k, topic=topic)

    elif args.command == "index_image":
        index_images(args.path)

    elif args.command == "search_image":
        search_images(args.query, top_k=args.top_k)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()