`timescale 1ns / 1ps
// =============================================================
// LENET5_TOP_STRUCTURAL
// Ghep noi 8 module lop rieng biet (conv1, pool1, conv2, pool2,
// fc1, fc2, fc3, argmax) thanh mot mach LeNet-5 hoan chinh.
// Chuc nang giong het lenet5_top.v ban dau, nhung duoc chia
// thanh cac khoi (module) rieng, giao tiep qua cong ghi/doc bo nho.
// =============================================================
module lenet5_top_structural (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output wire        ready,
    output reg          done,

    input  wire signed [7:0] pixel_in,
    input  wire        valid_in,

    output reg  [5:0]  predicted_class,
    output reg         valid_out
);

    // ---------------------------------------------------------
    // Tin hieu dieu khien / trang thai cua tung lop
    // ---------------------------------------------------------
    reg  c1_start, p1_start, c2_start, p2_start, f1_start, f2_start, f3_start, am_start;
    wire c1_done,  p1_done,  c2_done,  p2_done,  f1_done,  f2_done,  f3_done,  am_done;

    // ---------------------------------------------------------
    // Cong ghi dau vao cua tung lop
    // ---------------------------------------------------------
    reg         c1_in_wr_en; reg [9:0]  c1_in_wr_addr; reg signed [7:0] c1_in_wr_data;
    reg         p1_in_wr_en; reg [11:0] p1_in_wr_addr; reg signed [7:0] p1_in_wr_data;
    reg         c2_in_wr_en; reg [9:0]  c2_in_wr_addr; reg signed [7:0] c2_in_wr_data;
    reg         p2_in_wr_en; reg [10:0] p2_in_wr_addr; reg signed [7:0] p2_in_wr_data;
    reg         f1_in_wr_en; reg [8:0]  f1_in_wr_addr; reg signed [7:0] f1_in_wr_data;
    reg         f2_in_wr_en; reg [6:0]  f2_in_wr_addr; reg signed [7:0] f2_in_wr_data;
    reg         f3_in_wr_en; reg [6:0]  f3_in_wr_addr; reg signed [7:0] f3_in_wr_data;
    reg         am_in_wr_en; reg [5:0]  am_in_wr_addr; reg signed [31:0] am_in_wr_data;

    // ---------------------------------------------------------
    // Cong doc dau ra cua tung lop
    // ---------------------------------------------------------
    reg  [11:0] c1_out_rd_addr; wire signed [7:0]  c1_out_rd_data;
    reg  [9:0]  p1_out_rd_addr; wire signed [7:0]  p1_out_rd_data;
    reg  [10:0] c2_out_rd_addr; wire signed [7:0]  c2_out_rd_data;
    reg  [8:0]  p2_out_rd_addr; wire signed [7:0]  p2_out_rd_data;
    reg  [6:0]  f1_out_rd_addr; wire signed [7:0]  f1_out_rd_data;
    reg  [6:0]  f2_out_rd_addr; wire signed [7:0]  f2_out_rd_data;
    reg  [5:0]  f3_out_rd_addr; wire signed [31:0] f3_out_rd_data;

    conv1 u_conv1 (.clk(clk), .rst_n(rst_n), .start(c1_start), .ready(), .done(c1_done),
                   .in_wr_en(c1_in_wr_en), .in_wr_addr(c1_in_wr_addr), .in_wr_data(c1_in_wr_data),
                   .out_rd_addr(c1_out_rd_addr), .out_rd_data(c1_out_rd_data));

    pool1 u_pool1 (.clk(clk), .rst_n(rst_n), .start(p1_start), .ready(), .done(p1_done),
                   .in_wr_en(p1_in_wr_en), .in_wr_addr(p1_in_wr_addr), .in_wr_data(p1_in_wr_data),
                   .out_rd_addr(p1_out_rd_addr), .out_rd_data(p1_out_rd_data));

    conv2 u_conv2 (.clk(clk), .rst_n(rst_n), .start(c2_start), .ready(), .done(c2_done),
                   .in_wr_en(c2_in_wr_en), .in_wr_addr(c2_in_wr_addr), .in_wr_data(c2_in_wr_data),
                   .out_rd_addr(c2_out_rd_addr), .out_rd_data(c2_out_rd_data));

    pool2 u_pool2 (.clk(clk), .rst_n(rst_n), .start(p2_start), .ready(), .done(p2_done),
                   .in_wr_en(p2_in_wr_en), .in_wr_addr(p2_in_wr_addr), .in_wr_data(p2_in_wr_data),
                   .out_rd_addr(p2_out_rd_addr), .out_rd_data(p2_out_rd_data));

    fc1 u_fc1   (.clk(clk), .rst_n(rst_n), .start(f1_start), .ready(), .done(f1_done),
                   .in_wr_en(f1_in_wr_en), .in_wr_addr(f1_in_wr_addr), .in_wr_data(f1_in_wr_data),
                   .out_rd_addr(f1_out_rd_addr), .out_rd_data(f1_out_rd_data));

    fc2 u_fc2   (.clk(clk), .rst_n(rst_n), .start(f2_start), .ready(), .done(f2_done),
                   .in_wr_en(f2_in_wr_en), .in_wr_addr(f2_in_wr_addr), .in_wr_data(f2_in_wr_data),
                   .out_rd_addr(f2_out_rd_addr), .out_rd_data(f2_out_rd_data));

    fc3 u_fc3   (.clk(clk), .rst_n(rst_n), .start(f3_start), .ready(), .done(f3_done),
                   .in_wr_en(f3_in_wr_en), .in_wr_addr(f3_in_wr_addr), .in_wr_data(f3_in_wr_data),
                   .out_rd_addr(f3_out_rd_addr), .out_rd_data(f3_out_rd_data));

    wire [5:0] am_predicted_class;
    wire       am_valid_out;
    argmax u_argmax (.clk(clk), .rst_n(rst_n), .start(am_start), .ready(), .done(am_done),
                   .in_wr_en(am_in_wr_en), .in_wr_addr(am_in_wr_addr), .in_wr_data(am_in_wr_data),
                   .predicted_class(am_predicted_class), .valid_out(am_valid_out));

    // ---------------------------------------------------------
    // FSM dieu khien tong: nap anh -> chay tung lop -> copy du lieu
    // giua cac lop qua cong ghi/doc bo nho -> lay ket qua argmax
    // ---------------------------------------------------------
    localparam
        IDLE       = 0,
        LOAD       = 1,
        RUN_C1     = 2,
        CP_P1IN_A  = 3,  CP_P1IN_W  = 4,
        RUN_P1     = 5,
        CP_C2IN_A  = 6,  CP_C2IN_W  = 7,
        RUN_C2     = 8,
        CP_P2IN_A  = 9,  CP_P2IN_W  = 10,
        RUN_P2     = 11,
        CP_F1IN_A  = 12, CP_F1IN_W  = 13,
        RUN_F1     = 14,
        CP_F2IN_A  = 15, CP_F2IN_W  = 16,
        RUN_F2     = 17,
        CP_F3IN_A  = 18, CP_F3IN_W  = 19,
        RUN_F3     = 20,
        CP_AMIN_A  = 21, CP_AMIN_W  = 22,
        RUN_ARGMAX = 23,
        FINISH     = 24;

    reg [4:0] state;
    reg [15:0] cnt;
    reg [9:0]  pixel_cnt;

    assign ready = (state == IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done  <= 0;
            valid_out <= 0;
            predicted_class <= 0;
            {c1_start,p1_start,c2_start,p2_start,f1_start,f2_start,f3_start,am_start} <= 0;
            {c1_in_wr_en,p1_in_wr_en,c2_in_wr_en,p2_in_wr_en,f1_in_wr_en,f2_in_wr_en,f3_in_wr_en,am_in_wr_en} <= 0;
        end else begin
            done  <= 0;
            valid_out <= 0;
            c1_start <= 0; p1_start <= 0; c2_start <= 0; p2_start <= 0;
            f1_start <= 0; f2_start <= 0; f3_start <= 0; am_start <= 0;
            c1_in_wr_en <= 0; p1_in_wr_en <= 0; c2_in_wr_en <= 0; p2_in_wr_en <= 0;
            f1_in_wr_en <= 0; f2_in_wr_en <= 0; f3_in_wr_en <= 0; am_in_wr_en <= 0;

            case (state)
                IDLE: if (start) begin pixel_cnt <= 0; state <= LOAD; end

                // Nap 784 pixel truc tiep vao bo dem dau vao cua conv1
                LOAD: begin
                    if (valid_in) begin
                        c1_in_wr_en   <= 1;
                        c1_in_wr_addr <= pixel_cnt;
                        c1_in_wr_data <= pixel_in;
                        pixel_cnt <= pixel_cnt + 1;
                        if (pixel_cnt == 783) begin
                            c1_start <= 1;
                            state <= RUN_C1;
                        end
                    end
                end

                RUN_C1: if (c1_done) begin cnt <= 0; c1_out_rd_addr <= 0; state <= CP_P1IN_A; end

                // Copy conv1_out (4056) -> pool1 in
                CP_P1IN_A: begin c1_out_rd_addr <= cnt; state <= CP_P1IN_W; end
                CP_P1IN_W: begin
                    p1_in_wr_en   <= 1;
                    p1_in_wr_addr <= cnt;
                    p1_in_wr_data <= c1_out_rd_data;
                    if (cnt == 4055) begin p1_start <= 1; state <= RUN_P1; end
                    else begin cnt <= cnt + 1; state <= CP_P1IN_A; end
                end

                RUN_P1: if (p1_done) begin cnt <= 0; state <= CP_C2IN_A; end

                // Copy pool1_out (1014) -> conv2 in
                CP_C2IN_A: begin p1_out_rd_addr <= cnt; state <= CP_C2IN_W; end
                CP_C2IN_W: begin
                    c2_in_wr_en   <= 1;
                    c2_in_wr_addr <= cnt;
                    c2_in_wr_data <= p1_out_rd_data;
                    if (cnt == 1013) begin c2_start <= 1; state <= RUN_C2; end
                    else begin cnt <= cnt + 1; state <= CP_C2IN_A; end
                end

                RUN_C2: if (c2_done) begin cnt <= 0; state <= CP_P2IN_A; end

                // Copy conv2_out (1936) -> pool2 in
                CP_P2IN_A: begin c2_out_rd_addr <= cnt; state <= CP_P2IN_W; end
                CP_P2IN_W: begin
                    p2_in_wr_en   <= 1;
                    p2_in_wr_addr <= cnt;
                    p2_in_wr_data <= c2_out_rd_data;
                    if (cnt == 1935) begin p2_start <= 1; state <= RUN_P2; end
                    else begin cnt <= cnt + 1; state <= CP_P2IN_A; end
                end

                RUN_P2: if (p2_done) begin cnt <= 0; state <= CP_F1IN_A; end

                // Copy pool2_out (400) -> fc1 in
                CP_F1IN_A: begin p2_out_rd_addr <= cnt; state <= CP_F1IN_W; end
                CP_F1IN_W: begin
                    f1_in_wr_en   <= 1;
                    f1_in_wr_addr <= cnt;
                    f1_in_wr_data <= p2_out_rd_data;
                    if (cnt == 399) begin f1_start <= 1; state <= RUN_F1; end
                    else begin cnt <= cnt + 1; state <= CP_F1IN_A; end
                end

                RUN_F1: if (f1_done) begin cnt <= 0; state <= CP_F2IN_A; end

                // Copy fc1_out (120) -> fc2 in
                CP_F2IN_A: begin f1_out_rd_addr <= cnt; state <= CP_F2IN_W; end
                CP_F2IN_W: begin
                    f2_in_wr_en   <= 1;
                    f2_in_wr_addr <= cnt;
                    f2_in_wr_data <= f1_out_rd_data;
                    if (cnt == 119) begin f2_start <= 1; state <= RUN_F2; end
                    else begin cnt <= cnt + 1; state <= CP_F2IN_A; end
                end

                RUN_F2: if (f2_done) begin cnt <= 0; state <= CP_F3IN_A; end

                // Copy fc2_out (84) -> fc3 in
                CP_F3IN_A: begin f2_out_rd_addr <= cnt; state <= CP_F3IN_W; end
                CP_F3IN_W: begin
                    f3_in_wr_en   <= 1;
                    f3_in_wr_addr <= cnt;
                    f3_in_wr_data <= f2_out_rd_data;
                    if (cnt == 83) begin f3_start <= 1; state <= RUN_F3; end
                    else begin cnt <= cnt + 1; state <= CP_F3IN_A; end
                end

                RUN_F3: if (f3_done) begin cnt <= 0; state <= CP_AMIN_A; end

                // Copy fc3_out (47, 32-bit logit) -> argmax in
                CP_AMIN_A: begin f3_out_rd_addr <= cnt; state <= CP_AMIN_W; end
                CP_AMIN_W: begin
                    am_in_wr_en   <= 1;
                    am_in_wr_addr <= cnt;
                    am_in_wr_data <= f3_out_rd_data;
                    if (cnt == 46) begin am_start <= 1; state <= RUN_ARGMAX; end
                    else begin cnt <= cnt + 1; state <= CP_AMIN_A; end
                end

                RUN_ARGMAX: if (am_done) state <= FINISH;

                FINISH: begin
                    predicted_class <= am_predicted_class;
                    valid_out <= 1;
                    done <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule