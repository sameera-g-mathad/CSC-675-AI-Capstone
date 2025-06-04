import { KeyboardAvoidingView, Platform, SafeAreaView, StyleSheet, Text, View, Image } from "react-native";
import React, { useEffect, useContext } from "react";
import { useNavigation, useLocalSearchParams, useRouter } from "expo-router";
import ChatHistContext from "@/app/_context/ChathistContext";
import Chatbox from "@/app/_components/Chatbox";
import Content from "@/app/_components/Content";
import { HeaderBackButton } from '@react-navigation/elements';

export default function Chat() {
    const navigation = useNavigation();
    const router = useRouter()
    const { chat_title, displayImage } = useContext(ChatHistContext)
    const { chat_uuid } = useLocalSearchParams()
    useEffect(() => {
        navigation.setOptions({
            header: () =>
            (
                <View style={{
                    height: 70,
                    padding: 10,
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'center',
                    backgroundColor: '#6ee7b7'
                }}>
                    <HeaderBackButton onPress={() => router.replace('/')} />
                    {displayImage !== ''
                        &&
                        <Image
                            source={{ uri: `https://ai-capstone-s3-bucket.s3.us-east-2.amazonaws.com/nst/${displayImage}` }}
                            style={{
                                width: 40,
                                height: 40,
                                borderRadius: '50%',
                                marginRight: 10
                            }}
                            resizeMode="cover" />}
                    < Text >{chat_title}</ Text>
                </View>
            )
        })
    }, [chat_title])
    return (
        <SafeAreaView style={styles.viewContainer}>
            <KeyboardAvoidingView
                behavior={Platform.OS === "ios" ? "padding" : "height"}
                keyboardVerticalOffset={130}
                style={{ flex: 1 }}>
                <View style={styles.chatContainer}>
                    <Content uuid={typeof chat_uuid === 'string' ? chat_uuid : ''} />
                </View>
                <View>
                    <Chatbox uuid={typeof chat_uuid === 'string' ? chat_uuid : ''} />
                </View>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    viewContainer: {
        flex: 1,
    },
    chatContainer: {
        flex: 1,
    },

});
