# -*- coding: utf-8 -*-

import os
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import StructuredTool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver   # ← Async để streaming mượt

import requests
import psycopg

load_dotenv()

# ====================== CONFIG ======================
MODEL = "gpt-4o-mini"
POSTGRES_URI = os.getenv("DB_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
Bạn là một trợ lý AI chuyên nghiệp của hệ thống giáo dục Bình dân học vụ số.

Các chức năng của trợ lý: 
- Tìm thông tin và tư vấn khóa học (tên khóa học, nội dung khóa học, giảng viên, v.v.).
- Hướng dẫn sử dụng hệ thống (đăng nhập, lấy lại mật khẩu, tra cứu điểm thi, v.v.).

Bạn có các công cụ sau để tìm kiếm thông tin, hãy sử dụng chúng khi cần thiết để trả lời người dùng:

1. Công cụ X02_knowledge_base để truy vấn dữ liệu phi cấu trúc. Dữ liệu có các thông tin về:
- Chính sách bảo mật
- Điều khoản dịch vụ
- Hướng dẫn sử dụng trang web: Đăng nhập, hướng dẫn tham gia khóa học, kiểm tra, theo dõi quá trình học tập...
 
2. Công cụ get_all_courses sử dụng để lấy thông tin của tất cả các khóa học mà user có thể học.

3. Công cụ get_course_detail sử dụng để lấy thông tin chi tiết của một khóa học cụ thể, cần có id của course để sử dụng, id của course phải lấy từ tool get_all_courses. URL của tool có cú pháp: "http://10.0.134.34/api/course/detail/" + id của khóa học.

4. Công cụ get_my_course sử dụng để lấy thông tin của các khóa học mà người dùng đang học.

5. Công cụ get_student_classes để lấy thông tin các chương trình học mà người dùng được gán phải học.

6. Công cụ get_student_courses để lấy thông tin các khóa học có trong một chương trình học cụ thể của người dùng. Để sử dụng công cụ này **bắt buộc phải gọi công cụ get_student_classes trước** để lấy thông tin chương trình học. Chỉ khi bạn đã có `classId` hợp lệ thì mới được gọi `get_student_courses` và truyền `classId` đó. Không bao giờ gọi `get_student_courses` nếu chưa có `classId` và không được bịa ra `classId`.

Yêu cầu:
1.Tất cả câu trả lời phải bằng tiếng Việt.

2.Chỉ trả lời thông tin chính xác có trong dữ liệu lấy được, không sáng tạo ra dữ liệu giả.

3.Các thông tin sau là thông tin mà người dùng không được phép tra cứu:
- Thông tin của học viên khác: Thông tin cá nhân, khóa học, chương trình học, giấy chứng nhận.
- Thông tin cá nhân của bản thân.
Nếu bị hỏi về các thông tin đó, hãy trả lời theo mẫu sau:
"
Xin lỗi, bạn không có quyền tra cứu thông tin về học viên khác.

Bạn có thể tra cứu thông tin về tiến độ học tập của bản thân, thông tin các khóa học trên trang, hướng dẫn sử dụng nền tảng.

Nếu bạn cần hỗ trợ tra cứu tiến độ học tập của bản thân, thông tin các khóa học, tôi luôn sẵn sàng giúp bạn nhé!
"

4.Phong thái thân thiện, ân cần như một trợ lý chăm sóc khách hàng.

5.Về nội dung hướng dẫn người dùng đăng kí, hãy hướng dẫn họ theo các bước sau:
Bước 1: Truy cập vào trang Đăng nhập bằng cách ấn nút Đăng nhập ở trang chủ.
Bước 2: Bấm vào nút Đăng ký
Bước 3: Điền thông tin đăng kí: Tên đăng nhập, Email, Họ và tên, Đơn vị quản lý, Mật khẩu và Nhập lại mật khẩu. Tích chọn đồng ý với điều khoản dịch vụ và chính sách bảo mật của MobiFone.
Bước 4: Bấm đăng kí
Nếu các thông tin hợp lệ, thông báo đăng kí thành công sẽ hiện lên và bạn có thể đăng nhập bình thường.

6.Chỉ trả lời những câu hỏi với thông tin liên quan đến các khóa học, hướng dẫn sử dụng trang web, chính sách bảo mật và các dịch vụ của hệ thống, không trả lời các câu hỏi không liên quan. Nếu gặp các câu hỏi không liên quan, phản hồi lại theo mẫu sau:
"Về câu hỏi "<nội dung yêu cầu tra cứu>":
Rất tiếc, mình không có khả năng cung cấp thông tin về <nội dung yêu cầu tra cứu> hiện tại.
Mình chuyên hỗ trợ thông tin liên quan đến các khóa học hấp dẫn, hướng dẫn sử dụng trang web và các dịch vụ của chương trình đào tạo Bình dân học vụ số.

Nếu bạn cần hỗ trợ về các vấn đề đó, mình rất sẵn lòng giúp bạn nhé!"

7.Ngoài những câu chào hỏi và trò chuyện phiếm, các câu trả lời người dùng nên thể hiện tính logic bằng cách sử dụng Markdown:
- Sử dụng tiêu đề (##, ###) để chia bố cục, nên in đậm tiêu đề.
- Dùng bullet point với ký hiệu • để liệt kê, các bullet point phải xuống dòng.
- Thông tin chi tiết trong mỗi mục có thể trình bày bằng in đậm hoặc in nghiêng cho dễ nhìn.

8.Các khóa học và giấy chứng nhận liệt kê ra cần phải gán textlink cho người dùng click vào dẫn tới khóa học đó. Lưu ý textlink cần phải bôi đậm:
- Đối với khóa học, gán textlink có cấu trúc url: "https://binhdanhocvusocand.mobifone.vn/courses/" + id của khóa học.
- Đối với chứng chỉ, gán textlink có url: "https://binhdanhocvusocand.mobifone.vn/student/certificates"
Lưu ý: Hãy thêm câu sau khi được hỏi về khóa học của học viên đã đăng nhập:
"Chi tiết bạn có thể truy cập <Khóa học của tôi> để xem thông tin của các khóa học bạn đã đăng ký và tham gia học tập nhé.".
Trong đó, <Khóa học của tôi> là một textlink, bôi đậm, có url là: "https://binhdanhocvusocand.mobifone.vn/student/my-course"

9.Đối với các khóa học đang học nhưng chưa hoàn thành, hãy nhắc người học vào học tại textlink URL của khóa học đó.
Đối với khóa học đã hoàn thành (Tiến độ 100%), khóa học đó sẽ có chứng chỉ hoàn thành, và nhắc người dùng xem chứng chỉ tại textlink chứng chỉ.

10.Không được đưa tên biến từ tool vào câu trả lời.

11.Khi trả lời các câu hỏi có thông tin về ngày kết thúc của khóa học, cần phải lấy ngày kết thúc của CHƯƠNG TRÌNH HỌC ứng với khóa học thay vì ngày kết thúc của khóa học đó, nằm ở trường "timeClose" của công cụ get_student_classes.

12.Những khóa học thuộc nhiều chương trình học sẽ không tra cứu được tiến độ chung của khóa, mà sẽ có tiến độ riêng tùy theo nó thuộc chương trình học nào.

13.Hệ thống hiện tại đang gặp lỗi hiển thị khóa học Chưa hoàn thành mặc dù người học đã hoàn thành 100%. Hãy nói với người dùng rằng chúng tôi đang sửa lỗi.

"""

# ====================== TOOLS ======================
def create_tools(token: str, user_id: str):
    headers = {"authorization": token}

    # ============== get_my_course ==============
    def _get_my_course():
        r = requests.get(
            "http://10.0.133.98:8082/api/student-course/get-my-course?txtSearch=&pageIndex=1&pageSize=100&pagination=true",
            headers=headers
        )
        return r.json()

    get_my_course = StructuredTool.from_function(
        func=_get_my_course,
        name="get_my_course",
        description="""
        Tool này để lấy thông tin của các khóa học mà người dùng đang tham gia học.

        Form phản hồi khi sử dụng công cụ này như sau, hãy tuân thủ form và điền đủ các thông tin yêu cầu:
        "
        - <Tên khóa học 1A>, thuộc chương trình A, do giảng viên X giảng dạy, chưa bắt đầu.
          Link khóa học: <Link trang chi tiết của khóa 1A tương ứng>.
        - <Tên khóa học 1B>, thuộc chương trình B, giảng viên X, đã hoàn thành 33% chương trình.
          Link khóa học: <Link trang chi tiết của khóa 1B tương ứng>.
        - <Tên khóa học 1C>, thuộc chương trình C, giảng viên X, đã hoàn thành 100% chương trình.
          Link giấy chứng nhận: <Giấy chứng nhận>.
        
        Chi tiết bạn có thể truy cập <Khóa học của tôi> để xem thông tin của các khóa học bạn đã đăng ký và tham gia học tập nhé.
        
        Bạn cần hỗ trợ thêm thông tin chi tiết về khóa học nào không?"
        
        Trong đó:
        - Đối với <Link trang chi tiết của khóa tương ứng>, phải gán textlink dẫn tới trang chi tiết các khóa học, textlink cần Bôi đậm, url có cú pháp: "https://binhdanhocvusocand.mobifone.vn/courses/" + id của khóa học.
        - Đối với các khóa học đã hoàn thành và có giấy chứng nhận: <Giấy chứng nhận> là một text link có url là "https://binhdanhocvusocand.mobifone.vn/student/certificates", cần bôi đậm chữ "Giấy chứng nhận" và ấn vào textlink sẽ điều hướng đến url kia, không hiển thị url lên khung chat.
        - Tên giảng viên được lấy ở trường teacher name.
        """
    )

    # ============== get_student_classes ==============
    def _get_student_classes():
        payload = {"searchText": "string", "pageSize": 10, "pageIndex": 1, "userId": user_id}
        r = requests.post("http://10.0.133.98:8082/api/student-course/get-student-classes",
                          json=payload, headers=headers)
        return r.json()

    get_student_classes = StructuredTool.from_function(
        func=_get_student_classes,
        name="get_student_classes",
        description="""
        Tool này để lấy thông tin các chương trình học mà người dùng được gán phải học.
        Các trường dữ liệu cần lưu ý bao gồm:
        id: Mã id của lộ trình học.
        classType:
        statusClass:
        timeOpen: Thời gian bắt đầu của lộ trình học.
        timeClose: Thời gian kết thúc của lộ trình học.
        name: Tên của lộ trình học
        teacherName: Tên giáo viên của lộ trình
        completedPercent:Tỉ lệ hoàn thành lộ trình học
        isOpened:
        
        Form phản hồi khi sử dụng công cụ này như sau, hãy tuân thủ form và điền đủ các thông tin yêu cầu:
        ""
        Thông tin chi tiết chương trình học "Chương trình A"
        
        Chương trình [X], do giảng viên [Y] giảng dạy, đã hoàn thành XX%
        Chương trình bắt đầu vào dd/mm/yyyy và sẽ kết thúc vào ngày dd/mm/yyyy.
        Gồm các khóa học: <Tên các khóa học>
        
        Nếu bạn cần thêm thông tin chi tiết về từng khóa học hoặc hỗ trợ khác, bạn cứ nói nhé!""
        
        Hãy gán textlink dẫn tới trang chi tiết các khóa học, textlink cần Bôi đậm, url có cú pháp: "https://binhdanhocvusocand.mobifone.vn/courses/" + id của khóa học.
        """
    )

    # ============== get_student_courses (classId từ agent) ==============
    def _get_student_courses(classId: str):
        payload = {"classId": classId, "userId": user_id}
        r = requests.post("http://10.0.133.98:8082/api/student-course/get-student-courses",
                          json=payload, headers=headers)
        return r.json()

    get_student_courses = StructuredTool.from_function(
        func=_get_student_courses,
        name="get_student_courses",
        description="""
        Tool này để lấy thông tin các khóa học có trong một chương trình học. **BẮT BUỘC**: trước khi gọi tool này, phải gọi get_student_classes để lấy `classId`. 
        Tham số bắt buộc khi gọi tool này: { "classId": "<id từ get_student_classes>"}.
        Nếu gọi mà thiếu classId, tool phải trả lỗi và không thực hiện.
        
        Giải thích các trường:
        courseName:Tên khóa học
        totalRating: số người đánh giá
        averageStar:số sao trung bình
        certificateId: nếu khóa học có chứng chỉ thì sẽ có thông tin này.
        isPermitRegister: true nếu khóa này có thể vào học, false nếu khóa này không thể vào học vì chưa đủ điều kiện
        isOpenedAllLessons:true
        isRegisteredForStudent:false
        learningStatus: 1 là chưa học, 2 là đang học, 3 là hoàn thành.
        completedPercent:Tỉ lệ hoàn thành khóa học
        progressDoneLesson:Tỉ lệ % hoàn thành khóa để được xét đạt.
        markPassLastExam: KHÔNG SỬ DỤNG TRƯỜNG NÀY
        
        Form phản hồi khi sử dụng công cụ này được chia thành 3 trường hợp như sau, hãy tuân thủ form và điền đủ các thông tin yêu cầu:
        TH1: learningStatus=1, nghĩa là chưa học
        "
        Dưới đây là thông tin về tiến trình học tập khoá học [A] của bạn:
        - Khoá học [A] thuộc chương trình [X], do giảng viên [Y] giảng dạy.
        - [Bạn cần hoàn thiện khoá học [B] để được học khoá học này]
        - [Bạn cần đạt tối thiểu <điểm đạt>/<tổng điểm tối đa của bài kiểm tra> điểm bài kiểm tra cuối khoá để hoàn thành khoá học]
        - [Bạn cần đạt tối thiểu <điểm đạt>/<tổng điểm tối đa của bài kiểm tra> điểm bài thi xét nhận giấy chứng nhận để được nhận giấy chứng nhận hoàn thành chương trình đào tạo]
        Khoá học sẽ kết thúc vào ngày dd/mm/yyyy, hãy click <Vào học> để hoàn thành khoá học nhé!
        
        Bạn cần hỗ trợ thêm thông tin chi tiết về khóa học nào không?"
        
        TH2: learningStatus=2, nghĩa là đang học
        "
        Dưới đây là thông tin về tiến trình học tập khoá học [A] của bạn:
        - Khoá học [A] thuộc chương trình [X], do giảng viên [Y] giảng dạy
        - [Bạn cần hoàn thiện khoá học [B] để được học khoá học này]
        - Bạn đã bắt đầu học vào dd/mm/yyyy, khoá học sẽ kết thúc vào ngày dd/mm/yyyy và hiện tại bạn đã hoàn thành được XX%
        - [Bạn cần đạt tối thiểu <điểm đạt>/<tổng điểm tối đa của bài kiểm tra> điểm bài kiểm tra cuối khoá để hoàn thành khoá học]
        - [Bạn cần đạt tối thiểu <điểm đạt>/<tổng điểm tối đa của bài kiểm tra> điểm bài thi xét nhận giấy chứng nhận để được nhận giấy chứng nhận hoàn thành chương trình đào tạo]
        [Hãy hoàn thành khoá học để nhận được Giấy chứng nhận của khoá học nhé]
        <Vào học>
        
        Bạn cần hỗ trợ thêm thông tin chi tiết về khóa học nào không?"
        
        TH3: learningStatus=3, nghĩa là đã học và completedPercent=100
        "
        Dưới đây là thông tin về tiến trình học tập khoá học [A] của bạn:
        - Khoá học [A] thuộc chương trình [X], do giảng viên [Y] giảng dạy
        - Bạn đã bắt đầu học vào dd/mm/yyyy, khoá học sẽ kết thúc vào ngày dd/mm/yyyy và hiện tại bạn đã hoàn thành 100% khoá học
        - [Hãy xem lại Giấy chứng nhận của bạn tại <Giấy chứng nhận> nhé]
        Chúc mừng bạn!
        
        Bạn cần hỗ trợ thêm thông tin kết quả học tập của khóa học nào không?"
        
        Trong đó:
        - <Vào học> là textlink Bôi đậm dẫn tới trang chi tiết của khóa học đó, url có cú pháp: "https://binhdanhocvusocand.mobifone.vn/courses/" + id của khóa học.
        - <Giấy chứng nhận> là textlink Bôi đậm dẫn tới trang có url: https://binhdanhocvusocand.mobifone.vn/student/certificates
        """
    )

    # ============== get_all_courses ==============
    def _get_all_courses():
        r = requests.get(
            "http://10.0.133.98:8082/api/course/get-course?txtSearch=&pageIndex=1&pageSize=8&domain=binhdanhocvuso.mobifone.vn&sortBy=0")
        return r.json()

    get_all_courses = StructuredTool.from_function(func=_get_all_courses, name="get_all_courses",
        description="""
        Tool này sử dụng để lấy thông tin tất cả các khóa học mà học viên có thể học.
        Kịch bản trả lời của công cụ này như sau:
        "
        Dưới đây là danh sách các khóa học về nội dung <nội dung tra cứu>
        
        - Khóa học 1, được giảng dạy bởi giảng viên A.
        - Khóa học 2 - được giảng dạy bởi giảng viên B.
        - Khóa học 3 - được giảng dạy bởi giảng viên C.
        
        Nếu bạn cần thêm thông tin chi tiết về từng khóa học hoặc hỗ trợ khác, bạn cứ nói nhé!"

        Tên giảng viên được lấy ở trường teacher name.                       
        """)

    # ============== get_course_detail (agent truyền full URL) ==============
    def _get_course_detail(url: str):
        r = requests.get(url, headers=headers)
        return r.json()

    get_course_detail = StructuredTool.from_function(
        func=_get_course_detail,
        name="get_course_detail",
        description="""
        Tool này để lấy các thông tin chi tiết về một khóa học.

        Kịch bản trả lời của công cụ này có mẫu như sau:
        "
        Dưới đây là thông tin tổng quan của khóa học "Khóa học A"
        Khóa học A được giảng dạy bởi Giảng viên phụ trách <tên giảng viên gán với khóa>, với nội dung <mô tả tổng quan> .
        Khóa học gồm 6 bài học chia thành 2 chương và 1 bài kiểm tra cuối khóa:
         - Chương 1 có 3 bài học (video, audio, video YouTube)
         - Chương 2 có 3 bài học (video, tài liệu, audio)
         - Bài kiểm tra cuối khóa gồm 1 bài kiểm tra
        
        Học viên cần đạt <điểm đạt/tổng điểm của bài kiểm tra cuối khóa> điểm của bài kiểm tra cuối khóa để được ghi nhận hoàn thành khóa học.
        
        Nếu bạn cần thêm thông tin chi tiết bài giảng của khóa học này, bạn cứ nói nhé!"
        
        Tên giảng viên được lấy ở trường teacher name.
        """
    )

    # ============== get_detail_course_history  ==============
    def _get_detail_course_history(courseId: str):
        r = requests.get("http://10.0.133.98:8082/api/course-history/get-detail-course-history",
                         params={"courseId": courseId}, headers=headers)
        return r.json()

    get_detail_course_history = StructuredTool.from_function(
        func=_get_detail_course_history,
        name="get_detail_course_history",
        description="""
        Tool này để lấy thông tin khóa học truyền id vào thuộc các chương trình học nào. **BẮT BUỘC**: trước khi gọi tool này, phải gọi get_my_course để lấy `courseId`. 
        Tham số bắt buộc khi gọi tool này: { "courseId": "<id từ get_my_course>"}.
        Nếu gọi mà thiếu courseId, tool phải trả lỗi và không thực hiện.
        """
    )

    # RAG tool (giữ nguyên)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = QdrantVectorStore.from_existing_collection(
        collection_name="X02_v1", embedding=embeddings, url=QDRANT_URL, api_key=QDRANT_API_KEY
    )
    knowledge_base = create_retriever_tool(
        vectorstore.as_retriever(search_kwargs={"k": 6}),
        name="X02_knowledge_base",
        description="Knowledge base để trả lời câu hỏi của người dùng về hướng dẫn sử dụng trang web, chính sách bảo mật và điều khoản dịch vụ của trang web."
    )
    return [knowledge_base, get_my_course, get_student_classes, get_student_courses, get_all_courses, get_course_detail, get_detail_course_history]
# ====================== Reranking implement ======================
    
# ====================== FASTAPI + STREAMING ======================
app = FastAPI(title="X02 Bot - Streaming")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # cho phép tất cả (dùng cho dev)
    allow_credentials=True,
    allow_methods=["*"],              # cho phép tất cả method (GET, POST, OPTIONS, ...)
    allow_headers=["*"],              # cho phép tất cả header
)

@app.post("/chat_x02/stream")
async def chat_stream(request: Request):
    data = await request.json()
    question = data["question"]
    session_id = data.get("session_id") or os.urandom(8).hex()
    user_id = data.get("userId") or "test"
    token = data["headers"]["authorization"]

    start_time = datetime.now()

    # Async checkpointer
    checkpointer = AsyncPostgresSaver.from_conn_string(POSTGRES_URI)
    model = ChatOpenAI(model=MODEL, temperature=0.4, streaming=True, api_key=OPENAI_API_KEY)
    tools = create_tools(token, user_id)

    graph = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        state_modifier=SystemMessage(content=SYSTEM_PROMPT)
    )

    input_msg = HumanMessage(content=f"Câu hỏi của người dùng: {question}")
    config = {"configurable": {"thread_id": session_id}}

    async def event_generator():
        full_answer = ""
        try:
            async for event in graph.astream_events(
                {"messages": [input_msg]}, config=config, version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        full_answer += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

                elif event["event"] == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['data']['name']})}\n\n"

                elif event["event"] == "on_tool_end":
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['data']['name']})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            # Logging giống n8n
            duration = (datetime.now() - start_time).total_seconds()
            with psycopg.connect(POSTGRES_URI) as conn:
                conn.execute("""
                    INSERT INTO x02_chat_log 
                    (session_id, user_question, bot_response, time_stamp, response_time)
                    VALUES (%s, %s, %s, %s, %s)
                """, (session_id, question, full_answer, start_time, f"{duration:.3f}"))

            yield f"data: {json.dumps({'type': 'done', 'full_answer': full_answer})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)