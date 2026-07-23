`timescale 1ns / 1ps
// =============================================================
// POOL1 : Max-pooling 2x2, stride 2
// Vao : 6 x 26 x 26  (4056 gia tri, dia chi 12 bit)
// Ra  : 6 x 13 x 13  (1014 gia tri, dia chi 10 bit)
// =============================================================
module pool1 (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output wire        ready,
    output reg         done,

    // Cong ghi feature map dau vao (6 x 26 x 26 = 4056)
    input  wire        in_wr_en,
    input  wire [11:0] in_wr_addr,
    input  wire signed [7:0] in_wr_data,

    // Cong doc feature map dau ra (6 x 13 x 13 = 1014)
    input  wire [9:0]  out_rd_addr,
    output wire signed [7:0]  out_rd_data
);

    reg signed [7:0] c1_out [0:4055];
    reg signed [7:0] p1_out [0:1013];

    always @(posedge clk) begin
        if (in_wr_en) c1_out[in_wr_addr] <= in_wr_data;
    end
    assign out_rd_data = p1_out[out_rd_addr];

    localparam IDLE = 0, INIT = 1, COMP = 2, STORE = 3;
    reg [2:0] state;

    reg [15:0] c, y, x, ky, kx;
    reg signed [31:0] max_val;

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
                    max_val <= -128;
                    ky <= 0; kx <= 0;
                    state <= COMP;
                end

                COMP: begin
                    if (c1_out[c*676 + (y*2+ky)*26 + (x*2+kx)] > max_val)
                        max_val <= c1_out[c*676 + (y*2+ky)*26 + (x*2+kx)];

                    if (kx == 1) begin
                        kx <= 0;
                        if (ky == 1) state <= STORE;
                        else ky <= ky + 1;
                    end else kx <= kx + 1;
                end

                STORE: begin
                    p1_out[c*169 + y*13 + x] <= max_val[7:0];
                    if (x == 12) begin
                        x <= 0;
                        if (y == 12) begin
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