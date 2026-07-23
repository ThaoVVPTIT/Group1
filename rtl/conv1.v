`timescale 1ns / 1ps
// =============================================================
// CONV1 : Tich chap 3x3, 1 kenh vao -> 6 kenh ra
// Vao : 1 x 28 x 28   (784 gia tri, dia chi 10 bit)
// Ra  : 6 x 26 x 26   (4056 gia tri, dia chi 12 bit)
// Bao gom: MAC + Quant (nhan/dich) + Bias + ReLU + Clamp int8
// =============================================================
module conv1 (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output wire        ready,
    output reg         done,

    // Cong ghi anh dau vao (784 pixel, 1 kenh)
    input  wire        in_wr_en,
    input  wire [9:0]  in_wr_addr,
    input  wire signed [7:0] in_wr_data,

    // Cong doc feature map dau ra (6 x 26 x 26 = 4056)
    input  wire [11:0] out_rd_addr,
    output wire signed [7:0]  out_rd_data
);

    // ---------------------------------------------------------
    // 1. ROM trong so / bias / tham so luong tu hoa
    // ---------------------------------------------------------
    reg signed [7:0]  c1_kernel [0:53];   // 6 kenh * 9
    reg signed [31:0] c1_bias   [0:5];
    reg signed [31:0] c1_mult   [0:5];
    reg [7:0]         c1_shift  [0:5];

    initial begin
        $readmemh("weights_hex/conv1_kernel.hex",     c1_kernel);
        $readmemh("weights_hex/conv1_bias.hex",       c1_bias);
        $readmemh("weights_hex/conv1_multiplier.hex", c1_mult);
        $readmemh("weights_hex/conv1_shift.hex",      c1_shift);
    end

    // ---------------------------------------------------------
    // 2. RAM anh vao / feature map ra
    // ---------------------------------------------------------
    reg signed [7:0] img_buf [0:783];
    reg signed [7:0] c1_out  [0:4055];

    always @(posedge clk) begin
        if (in_wr_en) img_buf[in_wr_addr] <= in_wr_data;
    end
    assign out_rd_data = c1_out[out_rd_addr];

    // ---------------------------------------------------------
    // 3. FSM
    // ---------------------------------------------------------
    localparam IDLE = 0, INIT = 1, MAC = 2, QUANT = 3;
    reg [2:0] state;

    reg [15:0] c, y, x, ky, kx;
    reg signed [31:0] acc;
    reg signed [63:0] prod;

    assign ready = (state == IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done  <= 0;
        end else begin
            done <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        c <= 0; y <= 0; x <= 0;
                        state <= INIT;
                    end
                end

                INIT: begin
                    acc <= c1_bias[c];
                    ky <= 0; kx <= 0;
                    state <= MAC;
                end

                MAC: begin
                    acc <= acc + img_buf[(y+ky)*28 + (x+kx)] * c1_kernel[c*9 + ky*3 + kx];

                    if (kx == 2) begin
                        kx <= 0;
                        if (ky == 2) state <= QUANT;
                        else ky <= ky + 1;
                    end else kx <= kx + 1;
                end

                QUANT: begin
                    prod = acc * c1_mult[c];
                    if (c1_shift[c] > 0) begin
                        prod = prod + (64'sd1 << (c1_shift[c] - 1));
                        acc = prod >>> c1_shift[c];
                    end else acc = prod;

                    if (acc > 127) acc = 127; else if (acc < -128) acc = -128;
                    if (acc < 0) acc = 0; // ReLU

                    c1_out[c*676 + y*26 + x] <= acc[7:0];

                    if (x == 25) begin
                        x <= 0;
                        if (y == 25) begin
                            y <= 0;
                            if (c == 5) begin
                                done  <= 1;
                                state <= IDLE;
                            end else begin
                                c <= c + 1;
                                state <= INIT;
                            end
                        end else begin y <= y + 1; state <= INIT; end
                    end else begin x <= x + 1; state <= INIT; end
                end
            endcase
        end
    end
endmodule